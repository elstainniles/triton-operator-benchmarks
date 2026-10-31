import copy
import unittest

from opbench.liger_cv_large import load_case, selected, validate_merge, verify_all


class LigerLargeCVTests(unittest.TestCase):
    def test_all_frozen_launches_and_paired_measurements(self):
        self.assertEqual(verify_all(), dict(operators=6, applied_candidates=18, device_samples=1080))

    def test_explicit_suite_excludes_other_kernels_and_duplicates(self):
        self.assertEqual(len(selected('all')), 6)
        self.assertEqual([e['id'] for e, _, _ in selected('LK12,LK07')], ['LK12', 'LK07'])
        for value in ('LK46', 'LK07,LK07'):
            with self.assertRaises(ValueError):
                selected(value)

    def test_runtime_fallback_or_wrong_grid_cannot_be_reported_as_a_merge(self):
        _, _, case = load_case('LK07')
        launch = dict(runtime_status='APPLIED', applied_factor=8, selected_axis=0,
                      original_grid=case['grid'], batch_grid=[16, 4, 4])
        validate_merge(launch, 'static_masked_lanes_v1', 8, case['grid'])
        for field, value in [('runtime_status', 'FALLBACK'), ('applied_factor', 1),
                             ('selected_axis', 1), ('batch_grid', [128, 4, 4])]:
            broken = copy.deepcopy(launch)
            broken[field] = value
            with self.assertRaises(RuntimeError):
                validate_merge(broken, 'static_masked_lanes_v1', 8, case['grid'])
        with self.assertRaises(RuntimeError):
            validate_merge(launch, 'unknown', 8, case['grid'])


if __name__ == '__main__':
    unittest.main()
