"""Host-only checks; these do not compile or execute Triton on a device."""
import ast
import copy
import importlib.util
import json
import hashlib
import os
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(os.environ.get('FLA_GENERATED_ROOT', Path(__file__).resolve().parents[1]))
SYMBOLS = {'sigmoid_fwd_kernel', 'sigmoid_bwd_kernel', 'logsigmoid_fwd_kernel',
           'logsigmoid_bwd_kernel', 'softmax_fwd_kernel', 'softmax_bwd_kernel'}


def folders():
    if (ROOT / 'generated.json').is_file():
        return [ROOT / name for name in sorted(SYMBOLS)]
    return [p.parent for p in (ROOT / 'operators/flash_linear_attention').glob('*/operator.json')
            if json.loads(p.read_text()).get('entry_mode') == 'generated_benchmark']


def load(path, name='case'):
    package = '_fla_host_' + path.name
    if package not in sys.modules:
        module = types.ModuleType(package)
        module.__path__ = [str(path)]
        sys.modules[package] = module
    spec = importlib.util.spec_from_file_location(package + '.' + name, path / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def symbol_of(path):
    if (path / 'adapter-metadata.json').is_file():
        return json.loads((path / 'adapter-metadata.json').read_text())['symbol']
    return json.loads((path / 'operator.json').read_text())['kernel_symbol']


class GeneratedCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch is not None:
            torch.set_num_threads(2)
        cls.paths = folders()

    def test_six_exact_canonical_symbols(self):
        self.assertEqual({symbol_of(p) for p in self.paths}, SYMBOLS)
        self.assertEqual(len(self.paths), 6)

    @unittest.skipIf(torch is None, 'Optional CPU tensor checks require PyTorch')
    def test_all_configs_deterministic_masks_strides_guards(self):
        for path in self.paths:
            make_case = load(path).make_case
            for config in json.loads((path / 'cases.json').read_text())['cases']:
                with self.subTest(kernel=path.name, case=config['id']):
                    case = make_case(config, device='cpu')
                    again = make_case(config, device='cpu')
                    self.assertEqual(case.describe(), again.describe())
                    self.assertFalse(case.validate()['passed'], 'NaN outputs must fail before launch')
                    for output in case.outputs:
                        output['value'].copy_(output['expected'])
                    self.assertTrue(case.validate()['passed'])
                    for name, value in case.inputs.items():
                        if 'stride_' + name + '_row' in case.kwargs:
                            self.assertEqual(value.stride(0), config['shape'][1] + config.get('row_padding', {}).get(name, 0))
                            self.assertEqual(case.kwargs['stride_' + name + '_row'], value.stride(0))
                    case.outputs[0]['storage'][0] = 0
                    self.assertFalse(case.validate()['guards'])
                    case.prepare()
                    self.assertFalse(case.validate()['passed'])
                    for output in case.outputs:
                        output['value'].copy_(output['expected'])
                        if output['padding']:
                            output['storage'][16 + output['cols']] = 0
                            self.assertFalse(case.validate()['passed'])
                    case.prepare()
                    for output in case.outputs:
                        output['value'].copy_(output['expected'])
                    first_input = next(iter(case.input_storage.values()))[0]
                    first_input[-1] = 0
                    self.assertFalse(case.validate()['inputs_unchanged'])

    @unittest.skipIf(torch is None, 'Optional CPU tensor checks require PyTorch')
    def test_independent_math_and_gradients(self):
        for path in self.paths:
            symbol = symbol_of(path)
            config = json.loads((path / 'cases.json').read_text())['cases'][0]
            case = load(path).make_case(config, device='cpu')
            with self.subTest(kernel=symbol):
                if symbol.startswith('softmax'):
                    if '_bwd_' in symbol:
                        # Jacobian-vector product for normalized probabilities. Build the
                        # explicit Jacobian independently of the production reduction.
                        p = case.cpu_inputs['p'].double()
                        dp = case.cpu_inputs['dp'].double()
                        jac = torch.diag_embed(p) - p.unsqueeze(-1) * p.unsqueeze(-2)
                        expected = torch.einsum('nij,nj->ni', jac, dp)
                    else:
                        x = case.cpu_inputs['x'].double()
                        exponent = (x - x.max(-1, keepdim=True).values).exp()
                        expected = exponent / exponent.sum(-1, keepdim=True)
                else:
                    x = case.cpu_inputs['x'].double().clone().requires_grad_()
                    value = 1 / (1 + torch.exp(-x))
                    if symbol.startswith('logsigmoid'):
                        value = torch.log(value) / config['temperature']
                    expected = torch.autograd.grad(value, x, case.cpu_inputs['dy'].double())[0] if '_bwd_' in symbol else value
                torch.testing.assert_close(case.outputs[0]['expected'], expected, rtol=1e-12, atol=1e-12)

    @unittest.skipIf(torch is None, 'Optional CPU tensor checks require PyTorch')
    def test_invalid_layout_tile_dtype_temperature_rejected(self):
        for path in self.paths:
            make_case = load(path).make_case
            original = json.loads((path / 'cases.json').read_text())['cases'][0]
            bad = [dict(dtype='float16'), dict(shape=[0, 17]), dict(tile={'B': 123}),
                   dict(launch_options={'num_warps': 3, 'num_stages': 2}),
                   dict(launch_options={'num_warps': 4, 'num_stages': 3})]
            if symbol_of(path).startswith('softmax'):
                bad.append(dict(row_padding={'x': 1}))
            else:
                bad.append(dict(row_padding={'x': -1}))
            if symbol_of(path).startswith('logsigmoid'):
                bad.extend([dict(temperature=0), dict(temperature=float('nan'))])
            for update in bad:
                with self.subTest(kernel=path.name, invalid=update):
                    config = copy.deepcopy(original)
                    config.update(update)
                    with self.assertRaises(ValueError):
                        make_case(config, device='cpu')

    @unittest.skipIf(torch is None, 'Optional CPU tensor checks require PyTorch')
    def test_direct_launch_preserves_arguments_and_options(self):
        class FakeKernel:
            def __getitem__(self, grid):
                def call(**kwargs):
                    return grid, kwargs
                return call
        for path in self.paths:
            symbol = symbol_of(path)
            config = json.loads((path / 'cases.json').read_text())['cases'][0]
            case = load(path).make_case(config, device='cpu')
            kernels = types.SimpleNamespace(**{symbol: FakeKernel()})
            grid, kwargs = case.launch(kernels, symbol)
            self.assertEqual(grid, case.grid)
            for key, value in case.kwargs.items():
                self.assertIs(kwargs[key], value)
            self.assertIs(kwargs['enable_graph_optimize'], False)
            self.assertEqual(kwargs['num_warps'], 4)
            self.assertEqual(kwargs['num_stages'], 2)

    @unittest.skipIf(torch is None, 'Optional CPU tensor checks require PyTorch')
    def test_local_baseline_preserves_fixed_original_candidate(self):
        class FakeKernel:
            def __getitem__(self, grid):
                return lambda **kwargs: (grid, kwargs)
        for path in self.paths:
            symbol = symbol_of(path)
            config = json.loads((path / 'cases.json').read_text())['cases'][1]
            case = load(path).make_case(config, device='cpu')
            package = '_fla_host_' + path.name
            kernels = types.ModuleType(package + '.kernels')
            setattr(kernels, symbol, FakeKernel())
            sys.modules[kernels.__name__] = kernels
            setattr(sys.modules[package], 'kernels', kernels)
            baseline = load(path, 'baseline')
            grid, kwargs = baseline.launch(case)
            self.assertEqual(grid, case.grid)
            self.assertEqual(kwargs['num_warps'], 4)
            self.assertEqual(kwargs['num_stages'], 2)
            self.assertIs(kwargs['enable_graph_optimize'], False)
            if symbol.startswith('softmax'):
                self.assertEqual(kwargs['B'], 1 << (kwargs['D'] - 1).bit_length())
            else:
                self.assertEqual(kwargs['B'], 512)
                self.assertEqual(grid, ((case.kwargs['T'] + 511) // 512,))

    def test_no_manual_candidate_registered(self):
        for path in self.paths:
            manual = load(path, 'manual_tile')
            self.assertEqual(manual.CONFIGS, {})
            with self.assertRaisesRegex(ValueError, 'No manual candidate'):
                manual.launch(None, 'unregistered')

    def test_source_and_helper_changes_are_rejected(self):
        from opbench.fla import verify_operator
        for path in self.paths:
            verify_operator(path)
        source = next(p for p in self.paths if symbol_of(p) == 'sigmoid_fwd_kernel')
        for before, after, message in [
                ('mask=mask)', 'mask=True)', 'original signature, decorators, or body'),
                ("do_not_specialize=['T']", 'do_not_specialize=[]', 'original signature, decorators, or body'),
                ('return tl.exp(', 'return tl.exp2(', 'changes helper')]:
            with self.subTest(mutation=before), tempfile.TemporaryDirectory() as tmp:
                copied = Path(tmp) / 'operator'
                shutil.copytree(source, copied)
                kernel = copied / 'kernels.py'
                text = kernel.read_text()
                self.assertIn(before, text)
                kernel.write_text(text.replace(before, after, 1))
                # Re-hashing a mutation must not bypass the independent AST gate.
                metadata = json.loads((copied / 'operator.json').read_text())
                metadata['extraction']['sha256'] = hashlib.sha256(kernel.read_bytes()).hexdigest()
                (copied / 'operator.json').write_text(json.dumps(metadata))
                with self.assertRaisesRegex(ValueError, message):
                    verify_operator(copied)


if __name__ == '__main__':
    unittest.main()
