"""No-device checks for workbook identity, source binding and selection."""
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest

from opbench import fla_vv as vv
from opbench.fla_vv_runner import target_launch_policy, measure_rounds, select_input_config


class WorkbookCases(unittest.TestCase):
    def test_full_frozen_order_and_integrity(self):
        result = vv.verify_all()
        self.assertEqual(result['rows'], 317)
        self.assertEqual(result['configurations'], 318)
        self.assertEqual(result['operator_entries'], 130)
        self.assertEqual(result['device_validation'], 'NOT_RUN')

    def test_ranges_inclusive_ordered_and_deduplicated(self):
        rows = vv.selected('20,1-3,2,317')
        self.assertEqual([r['index'] for r in rows], [1, 2, 3, 20, 317])
        self.assertEqual([r['row'] for r in rows], [203, 204, 205, 222, 519])
        self.assertEqual(vv.selected('1-20'), vv.selected(rows='203-222'))

    def test_invalid_or_empty_selection_is_rejected(self):
        for value in ('0', '318', '20-1', 'all,1', '', '1,,2', '-1', '1-9999999999'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                vv.selected(value)
        with self.assertRaises(ValueError):
            vv.selected('1-3', rows='203-205')
        with self.assertRaises(ValueError):
            vv.selected(operator='FLA999999')

    def case(self, row):
        return vv.load_case(vv.selected(rows=str(row))[0])

    def test_function_parameters_and_tail_shapes_are_not_collapsed(self):
        for row, expected in ((318, False), (359, False), (361, True)):
            case = self.case(row)
            self.assertIs(case['configurations'][0]['invocation']['kwargs']['save_kl'], expected)
        for row, suffix in ((382, 'copy_input'), (383, 'copy_input_tail'), (389, 'cumsum_local_tail')):
            self.assertTrue(self.case(row)['configurations'][0]['invocation']['kwargs']['load_dir'].endswith(suffix))

    def test_unrecorded_warps_keep_both_existing_variants(self):
        case = self.case(488)
        self.assertIsNone(case['table_config']['num_warps'])
        self.assertEqual({c['config']['num_warps'] for c in case['configurations']}, {4, 8})
        self.assertTrue(all(c['correctness_kind'] == 'requires_acceptance_review' for c in case['configurations']))

    def test_multi_shape_original_test_is_restricted_to_workbook_row(self):
        first = self.case(220)['configurations'][0]
        third = self.case(223)['configurations'][0]
        self.assertEqual(first['test_id'], third['test_id'])
        self.assertEqual(first['input_selection']['index'], 0)
        self.assertEqual(third['input_selection']['index'], 2)
        self.assertEqual(third['input_selection']['expected']['temperature'], 2)
        module = SimpleNamespace(CONFIGS=[{'N':32}, {'N':64}])
        select_input_config(module, {'global_name':'CONFIGS', 'index':1, 'expected':{'N':64}})
        self.assertEqual(module.CONFIGS, [{'N':64}])
        with self.assertRaises(ValueError):
            select_input_config(module, {'global_name':'CONFIGS', 'index':0, 'expected':{'N':32}})

    def test_config_and_manifest_mutations_are_rejected(self):
        case = self.case(203)
        config = deepcopy(case['configurations'][0])
        config['config']['num_warps'] = 1
        with self.assertRaisesRegex(ValueError, 'configuration differs'):
            vv.load_test(case, config)
        config = deepcopy(case['configurations'][0])
        config['test_manifest_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'manifest changed'):
            vv.load_test(case, config)

    def test_planning_has_no_torch_or_triton_dependency(self):
        root = Path(__file__).resolve().parents[1]
        code = ('import sys; from opbench.fla_vv import selected, load_case; '
                'load_case(selected("1")[0]); '
                'assert "torch" not in sys.modules and "triton" not in sys.modules')
        subprocess.run([sys.executable, '-c', code], cwd=root, check=True)


class ControlledLaunch(unittest.TestCase):
    def test_pinned_positional_tile_and_no_inherited_block_options(self):
        jit = SimpleNamespace(arg_names=['x', 'N', 'BLOCK'])
        values, kw = target_launch_policy(jit, ('tensor', 100, 32),
            {'num_warps': 8, 'block_batching_factor': 8, 'block_batching_auto': True},
            {'BLOCK': 64, 'num_warps': 4})
        self.assertEqual(values, ('tensor', 100, 64))
        self.assertEqual(kw, {'num_warps': 4})

    def test_keyword_tile_retains_unrelated_launch_options(self):
        jit = SimpleNamespace(arg_names=['x', 'BLOCK'])
        values, kw = target_launch_policy(jit, ('tensor',),
            {'BLOCK': 32, 'enable_graph_optimize': False}, {'kwargs': {'BLOCK': 64}, 'num_warps': 4})
        self.assertEqual(kw, {'BLOCK': 64, 'num_warps': 4, 'enable_graph_optimize': False})

    def test_round_aggregation_is_mean_of_medians(self):
        import tempfile
        readings = iter(([1, 2, 900], [3, 4, 800]))
        def measure(*args):
            return {'samples_us': next(readings)}
        with tempfile.TemporaryDirectory() as directory:
            result = measure_rounds(measure, None, Path(directory), 3, 1, 2)
        self.assertEqual(result['kernel_us'], 3)
        self.assertNotEqual(result['mean_us'], result['kernel_us'])
        self.assertEqual(len(result['rounds']), 2)


if __name__ == '__main__':
    unittest.main()
