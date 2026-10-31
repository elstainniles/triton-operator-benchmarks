import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from opbench.liger_cv_large import launch_options, load_case, run_batch, selected
from opbench.liger_cv_small import LOWERING, validate_compiled, verify_all

ROOT = Path(__file__).resolve().parents[1]


class LigerSmallCVTests(unittest.TestCase):
    def test_frozen_original_inputs_and_all_historical_measurements(self):
        self.assertEqual(verify_all(), dict(operators=6, applied_candidates=18,
                                           confirmed_gain_candidates=8, guard_probes=12, device_samples=1920))

    def test_small_cli_and_large_default_select_different_shapes(self):
        output = subprocess.check_output([sys.executable, '-I', str(ROOT / 'tools/run_liger_cv_small.py'),
                                          'plan', '--operator', 'LK12,LK07'], text=True)
        rows = json.loads(output)
        self.assertEqual([r['id'] for r in rows], ['LK12', 'LK07'])
        self.assertTrue(all(r['case']['shape'] == [2, 4, 32, 32] for r in rows))
        self.assertEqual(load_case('LK07')[2]['shape'], [8, 16, 128, 64])
        self.assertEqual(len(selected('all', 'small')), 6)
        for value in ('LK46', 'LK07,LK07'):
            with self.assertRaises(ValueError):
                selected(value, 'small')

    def test_baseline_control_and_merge_flags_are_distinct(self):
        self.assertEqual(launch_options(1, 'small'), dict(enable_hivm_batch_matmul=False, debug=False))
        self.assertEqual(launch_options(1, 'small', True), dict(enable_hivm_batch_matmul=True, debug=False))
        for factor in (2, 4, 8):
            expected = dict(enable_hivm_batch_matmul=True, debug=False, block_batching_cv=True,
                            block_batching_factor=factor, block_batching_axis=0)
            self.assertEqual(launch_options(factor, 'small'), expected)
            expected['enable_hivm_batch_matmul'] = False
            self.assertEqual(launch_options(factor), expected)
        for factor, suite in ((2, 'small'), (1, 'large')):
            with self.assertRaises(ValueError):
                launch_options(factor, suite, True)

    def test_small_batch_dispatch_preserves_control_and_lock_arguments(self):
        commands = []

        def worker(command, **kwargs):
            commands.append(command)
            folder = Path(command[command.index('--output') + 1])
            folder.mkdir()
            (folder / 'result.json').write_text(json.dumps(dict(status='correct', median_us=None)))
            return SimpleNamespace(returncode=0)

        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            args = SimpleNamespace(operator='LK12,LK07', suite='small', factor=1, batch_control=True,
                                   physical_card=6, environment=base / 'environment.json', output=base / 'run',
                                   correctness_only=True, task_lock=base / 'task.lock', card_lock=base / 'card.lock')
            with patch('opbench.liger_cv_large.subprocess.run', side_effect=worker):
                run_batch(args)
            self.assertEqual(len(commands), 2)
            for command in commands:
                self.assertEqual(Path(command[2]).name, 'run_liger_cv_small.py')
                self.assertIn('--batch-control', command)
                self.assertIn('--correctness-only', command)
                for flag, expected in (('--task-lock', args.task_lock), ('--card-lock', args.card_lock)):
                    self.assertEqual(command[command.index(flag) + 1], str(expected.resolve()))
            self.assertTrue(json.loads((args.output / 'summary.json').read_text())['completed'])

    def test_fallback_or_wrong_backend_cannot_masquerade_as_batch_merge(self):
        compiled = SimpleNamespace(
            last_launch=dict(runtime_status='APPLIED', applied_factor=2, selected_axis=0,
                             original_grid=[8, 1, 1], batch_grid=[4, 1, 1]),
            block_batching=dict(cv_lowering=LOWERING, guard=dict(require_full_masks=True)),
            candidate=SimpleNamespace(
                metadata=SimpleNamespace(enable_hivm_batch_matmul=True, auto_blockify_enabled=False,
                                         default_dot_input_precision='ieee'),
                kernel=b'test binary',
                asm=dict(ttir='%c = tt.dot %a, %b, %z : tensor<2x32x32xf32> * '
                              'tensor<2x32x32xf32> -> tensor<2x32x32xf32>')))
        self.assertEqual(validate_compiled(compiled, 2, [8, 1, 1])['cv_lowering'], LOWERING)
        for field, value in (('runtime_status', 'FALLBACK'), ('applied_factor', 1),
                             ('selected_axis', 1), ('batch_grid', [8, 1, 1]), ('original_grid', [16, 1, 1])):
            broken = copy.deepcopy(compiled)
            broken.last_launch[field] = value
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                validate_compiled(broken, 2, [8, 1, 1])
        for field, value in (('enable_hivm_batch_matmul', False), ('auto_blockify_enabled', True),
                             ('default_dot_input_precision', 'hf32')):
            broken = copy.deepcopy(compiled)
            setattr(broken.candidate.metadata, field, value)
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                validate_compiled(broken, 2, [8, 1, 1])
        for kind in ('old_lowering', 'unproven_mask', 'rank2_dot'):
            broken = copy.deepcopy(compiled)
            if kind == 'old_lowering':
                broken.block_batching['cv_lowering'] = 'static_masked_lanes_v1'
            elif kind == 'unproven_mask':
                broken.block_batching['guard']['require_full_masks'] = False
            else:
                broken.candidate.asm['ttir'] = '%c = tt.dot %a, %b, %z : tensor<32x32xf32>'
            with self.subTest(kind=kind), self.assertRaises(RuntimeError):
                validate_compiled(broken, 2, [8, 1, 1])


if __name__ == '__main__':
    unittest.main()
