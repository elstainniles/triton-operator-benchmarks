import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from opbench.catalog import ROOT
from opbench.fla_cv import load_case, plan, selected, verify_all, verify_case
from opbench.fla_cv_reference import grid
from opbench.fla_cv_runner import run_batch


class FLACVTests(unittest.TestCase):
    def test_frozen_scope_and_historical_samples(self):
        self.assertEqual(verify_all(), dict(rows=201, operators=150, source_backed_rows=194,
                         historical_counts=dict(CONTROL_FLOW_BLOCKED=152, MEASURED=41,
                                                BASELINE_ACCURACY_FAILED=1, SOURCE_NOT_FOUND=7)))

    def test_selectors_do_not_silently_substitute_missing_source(self):
        rows = selected('24,26,37,120,201')
        self.assertEqual([r['row'] for r in rows], [24,26,37,120,201])
        self.assertIsNone(load_case(rows[2])[1])
        self.assertEqual([r['row'] for r in selected('tested')],
                         [r['row'] for r in selected() if r['historical_status'] not in
                          ('CONTROL_FLOW_BLOCKED','SOURCE_NOT_FOUND')])
        with self.assertRaisesRegex(ValueError,'Unknown workbook row'):
            selected('203')
        statuses = [r['status'] for r in plan(SimpleNamespace(cases='24,37,38',operator='all'))]
        self.assertEqual(statuses, ['ready','source_not_found','control_flow_not_revalidated'])

    def test_source_hash_and_kernel_body_cannot_drift(self):
        row = selected('120')[0]
        path, spec = load_case(row)
        self.assertTrue(verify_case(path,spec)['module_sha256'])
        copied = dict(spec,module_sha256='0'*64)
        with self.assertRaisesRegex(ValueError,'executable source hash'):
            verify_case(path,copied)
        copied = dict(spec,bodies=[dict(spec['bodies'][0],body_sha256='0'*64)])
        with self.assertRaisesRegex(ValueError,'function body changed'):
            verify_case(path,copied)

    def test_historical_executable_grids_match_frozen_cases(self):
        for row in selected('tested'):
            path, spec = load_case(row)
            with self.subTest(row=row['row']):
                actual = grid(spec, SimpleNamespace(__file__=path/spec['module']))
                self.assertEqual(list(actual), spec['grid'])

    def test_host_imports_do_not_load_accelerators(self):
        cmd = ('import sys;sys.path.insert(0,'+repr(str(ROOT))+');'
               'import opbench.fla_cv,opbench.fla_cv_runner,opbench.fla_cv_reference;'
               'assert "torch" not in sys.modules and "triton" not in sys.modules')
        subprocess.run([sys.executable,'-I','-c',cmd],check=True)

    def test_batch_prechecks_and_resumes_without_worker(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            args=SimpleNamespace(cases='37,38',operator='all',output=root/'batch',
                                 physical_card=6,factor=2,axis='all',resume=False,
                                 include_control_flow=False,correctness_only=False,
                                 environment=root/'environment.json',task_lock=None,card_lock=None,
                                 reference_timeout=600,worker_timeout=1800)
            with patch('opbench.fla_cv_runner.subprocess.run') as worker:
                run_batch(args)
                worker.assert_not_called()
            summary=json.loads((args.output/'summary.json').read_text())
            self.assertTrue(summary['completed'])
            self.assertEqual([summary['rows'][key]['status'] for key in ('37','38')],
                             ['SOURCE_NOT_FOUND','CONTROL_FLOW_BLOCKED_PRECHECK_ONLY'])
            args.resume=True
            with patch('opbench.fla_cv_runner.subprocess.run') as worker:
                run_batch(args)
                worker.assert_not_called()
            args.factor=4
            with self.assertRaisesRegex(ValueError,'original environment, locks, card and run options'):
                run_batch(args)
            args.factor=2
            args.cases='37,39'
            with self.assertRaisesRegex(ValueError,'same frozen selection'):
                run_batch(args)

    def test_compile_failure_is_a_resumable_terminal_result(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            args=SimpleNamespace(cases='201',operator='all',output=root/'batch',
                                 physical_card=6,factor=2,axis='all',resume=False,
                                 include_control_flow=False,correctness_only=False,
                                 environment=root/'environment.json',task_lock=None,card_lock=None,
                                 reference_timeout=600,worker_timeout=1800)

            def fake_worker(command, **kwargs):
                attempt=Path(command[command.index('--output')+1])
                attempt.mkdir()
                (attempt/'result.json').write_text(json.dumps(dict(
                    status='BASELINE_COMPILE_FAILED',baseline_us=None,candidate_us=None)))
                return SimpleNamespace(returncode=0)

            with patch('opbench.fla_cv_runner.subprocess.run',side_effect=fake_worker):
                run_batch(args)
            summary_path=args.output/'summary.json'
            summary=json.loads(summary_path.read_text())
            self.assertEqual(summary['rows']['201']['status'],'BASELINE_COMPILE_FAILED')
            del summary['rows']['201']
            summary['completed']=False
            summary_path.write_text(json.dumps(summary))
            args.resume=True
            with patch('opbench.fla_cv_runner.subprocess.run') as worker:
                run_batch(args)
                worker.assert_not_called()
            self.assertEqual(json.loads(summary_path.read_text())['rows']['201']['status'],
                             'BASELINE_COMPILE_FAILED')


if __name__ == '__main__':
    unittest.main()
