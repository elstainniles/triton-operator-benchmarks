"""Host contract tests: source fidelity, exact bindings and in-place timing safety."""
import ast
from contextlib import contextmanager
import inspect
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from opbench.catalog import ROOT, module, plans, read_json, resolve, select
from opbench.runner import prepare_case, profile_call
from opbench.tencent import verify_suite


class TencentTests(unittest.TestCase):
    def test_original_symbols_and_separate_collection(self):
        result = verify_suite()
        self.assertEqual(result['operators'], 6)
        self.assertEqual(select('Tencent'), select('tencent'))
        self.assertEqual([len(select(s)) for s in ('flaggems', 'graph', 'liger', 'fla', 'Tencent')],
                         [8, 7, 46, 296, 6])
        self.assertEqual([p['case']['shape'] for p in plans('Tencent')],
                         [[4096, 163840]] * 2 + [[2048, 512]] * 2 + [[128, 1024]] * 2)
        self.assertEqual([p['grid'] for p in plans('Tencent')],
                         [[4096, 80], [4096, 80], [2048], [64], [128, 8], [128, 8]])
        self.assertEqual(len(plans('Tencent', 'fp32')), 2)
        self.assertEqual(len(plans('Tencent', 'bf16')), 4)
        self.assertEqual(plans('Tencent', 'smoke'), plans('Tencent', 'all'))
        with self.assertRaisesRegex(ValueError, 'original baselines only'):
            plans('Tencent', variants='manual')

    def test_each_launch_matches_original_signature_and_inplace_contract(self):
        for entry in select('Tencent'):
            with self.subTest(operator=entry['id']):
                _, path = resolve(entry['id'])
                metadata = read_json(path / 'operator.json')
                tree = ast.parse((path / 'kernels.py').read_text(encoding='utf-8'))
                function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                                and n.name == metadata['kernel_symbol'])
                signature = inspect.Signature([inspect.Parameter(a.arg, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                                               for a in function.args.args])
                bindings = []

                class Kernel:
                    def __getitem__(self, launch_grid):
                        def call(*args, **kwargs):
                            bindings.append((launch_grid, signature.bind(*args, **kwargs).arguments))
                        return call

                case = SimpleNamespace(inputs={key: object() for key in
                    ('logits', 'labels', 'row_max', 'sum_exp', 'dloss', 'x', 'weight', 'dy', 'rrms', 'a', 'b', 'dout')},
                    actual=[object(), object()])
                fake_name = entry['path'].replace('/', '.') + '.kernels'
                kernel = SimpleNamespace(**{metadata['kernel_symbol']: Kernel()})
                planned, = plans(entry['id'])
                for shape in (planned['case']['shape'], (7, 2051)):
                    case.inputs['shape'] = shape
                    with patch.dict(sys.modules, {fake_name: kernel}):
                        module(entry, 'launch').launch(case, planned['config'])
                    actual_grid, bound = bindings[-1]
                    self.assertEqual(list(actual_grid), list(module(entry, 'launch').grid(shape, planned['config'])))
                    if entry['id'] == 'TX02':
                        self.assertIs(bound['dlogits_ptr'], bound['logits_ptr'])
                        self.assertIsNot(bound['logits_ptr'], case.inputs['logits'])
                        self.assertIs(bound['STORE_BF16'], False)
                    elif entry['id'] == 'TX03':
                        self.assertEqual(bound['eps'], 10.0)
                    elif entry['id'] == 'TX04':
                        self.assertEqual(bound['BLOCK_S'], (shape[0] + 63) // 64)
                        self.assertGreaterEqual(bound['D_ROUNDED'], shape[1])

    def test_reset_and_sync_stay_outside_every_timed_call(self):
        events = []
        state = {'logits': 7}

        def prepare():
            events.append('prepare')
            state['logits'] = 7

        def launch():
            self.assertEqual(state['logits'], 7)
            events.append('kernel')
            state['logits'] = -1

        @contextmanager
        def label(_):
            events.append('enter')
            yield
            events.append('exit')

        fake_torch = SimpleNamespace(npu=SimpleNamespace(synchronize=lambda: events.append('sync')),
                                     profiler=SimpleNamespace(record_function=label))
        case = SimpleNamespace(prepare=prepare)
        for _ in range(15):
            profile_call(fake_torch, case, launch, 'opbench::baseline::0::0')
        self.assertEqual(events, ['prepare', 'sync', 'enter', 'kernel', 'sync', 'exit'] * 15)
        events.clear()
        state['logits'] = 7
        profile_call(fake_torch, SimpleNamespace(), launch, 'legacy')
        self.assertEqual(events, ['sync', 'enter', 'kernel', 'exit', 'sync'])

    def test_unprepared_old_cases_keep_existing_initialization(self):
        events = []
        case = SimpleNamespace(actual=[SimpleNamespace(fill_=lambda value: events.append(value))])
        prepare_case(case, initial=True)
        self.assertEqual(len(events), 1)
        self.assertNotEqual(events[0], events[0])  # NaN prefill
        prepare_case(case)
        self.assertEqual(len(events), 1)

    def test_plan_does_not_import_accelerator_packages(self):
        code = ('import sys; from opbench.catalog import plans; '
                'assert len(plans("Tencent")) == 6; '
                'assert "torch" not in sys.modules and "triton" not in sys.modules')
        subprocess.run([sys.executable, '-c', code], cwd=ROOT, check=True, capture_output=True, text=True)

    def test_unsupported_timing_rejected_before_batch_execution(self):
        from opbench.cli import main
        argv = ['run.py', 'run', '--operator', 'all', '--timing', 'mspti', '--output', 'unused']
        with patch.object(sys, 'argv', argv), patch('opbench.batch.run') as batch:
            with self.assertRaises(SystemExit) as error:
                main()
        self.assertEqual(error.exception.code, 2)
        batch.assert_not_called()


if __name__ == '__main__':
    unittest.main()
