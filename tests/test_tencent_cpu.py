"""Optional CPU-only fixture checks; these are NOT Triton/NPU execution tests."""
import importlib.util
import unittest

from opbench.catalog import module, plans, select


@unittest.skipUnless(importlib.util.find_spec('torch'), 'CPU fixture checks require PyTorch')
class TencentCpuTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        torch.set_num_threads(4)

    def make(self, identifier, shape=None):
        entry = next(e for e in select('Tencent') if e['id'] == identifier)
        plan, = plans(identifier)
        config = plan['case']
        return module(entry, 'case').make_case(shape or config['shape'],
                                             getattr(self.torch, config['dtype']), 42, device='cpu')

    def test_ce_oracles_agree_with_native_loss_and_autograd(self):
        torch = self.torch
        fwd, bwd = self.make('TX01', (7, 2051)), self.make('TX02', (7, 2051))
        x = bwd.inputs['logits'].double().requires_grad_(True)
        labels = bwd.inputs['labels']
        loss = torch.nn.functional.cross_entropy(x, labels, reduction='none', ignore_index=-1)
        loss.backward(bwd.inputs['dloss'].double())
        expected_loss = next(check[2] for check in fwd.checks if check[0] == 'loss')
        torch.testing.assert_close(expected_loss, loss.detach(), rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(bwd.checks[0][2], x.grad.float(), rtol=1e-6, atol=1e-7)
        self.assertEqual(float(x.grad[0].abs().sum()), 0.0)

    def test_inplace_reset_recovers_inputs_after_every_gradient(self):
        torch = self.torch
        case = self.make('TX02', (7, 2051))
        original, target = case.inputs['logits'], case.actual[0]
        self.assertNotEqual(original.data_ptr(), target.data_ptr())
        for _ in range(3):
            case.prepare()
            torch.testing.assert_close(target, original, rtol=0, atol=0)
            target.copy_(case.checks[0][2])
            self.assertTrue(case.validate()['passed'])
        self.assertTrue(case.compare('source', original, case.readonly[0][2], 0, 0)['passed'])

    def test_six_mixed_dtype_fixtures_check_unwritten_outputs_and_guards(self):
        torch = self.torch
        for entry in select('Tencent'):
            with self.subTest(operator=entry['id']):
                # Exercise exact requested RMSNorm/SiTU sizes, but keep CE CPU checks small.
                case = self.make(entry['id'], (7, 2051) if entry['id'] in ('TX01', 'TX02') else None)
                case.prepare()
                self.assertFalse(case.validate()['passed'])
                for _, actual, expected, _, _ in case.checks:
                    if not callable(actual):
                        actual.copy_(expected.to(actual.dtype))
                self.assertTrue(case.validate()['passed'])
                storage = case.guards[0][1]
                storage[0] = 0
                self.assertFalse(case.validate()['guards'])

    def test_readonly_input_mutation_is_rejected(self):
        case = self.make('TX03', (17, 129))
        for _, actual, expected, _, _ in case.checks:
            if not callable(actual):
                actual.copy_(expected.to(actual.dtype))
        self.assertTrue(case.validate()['passed'])
        case.inputs['x'][0, 0] += 1
        result = case.validate()
        self.assertFalse(result['passed'])
        self.assertFalse(result['inputs'][0]['passed'])


if __name__ == '__main__':
    unittest.main()
