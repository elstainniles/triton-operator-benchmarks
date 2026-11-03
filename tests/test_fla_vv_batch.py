"""Batch state-machine tests using controlled child results, without a device."""
from copy import deepcopy
import csv
import subprocess
import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from opbench import fla_vv_runner as batch
from opbench.runner import save


class VvBatch(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.output = Path(temp.name)
        self.args = SimpleNamespace(factor=2)
        self.rows = [dict(index=i, row=i+202, id=f'vv-row-{i+202}', operator_id='FLA001',
                          kernel='kernel') for i in (1, 2)]
        self.config = dict(test_id='vv-example', config={'BLOCK': 32})
        self.summary = dict(run_config={'factor': 2}, trials={}, completed=False)
        self.calls = []
        self.addCleanup(patch.stopall)
        patch.object(batch, 'load_case', side_effect=lambda row: dict(shape_dtype='[100]/f32',
                     configurations=[deepcopy(self.config)])).start()
        patch.object(batch, 'load_test', return_value=(None, {})).start()

    def launch(self, outcomes):
        outcomes = iter(outcomes)
        def fake(args, row, config, variant, dest, fds):
            self.calls.append((row['row'], variant))
            status = next(outcomes)
            value = dict(status=status, timing=dict(kernel_us=4 if variant == 'baseline' else 2,
                performance_eligible=True) if status == 'measured' else None, completed=True)
            dest.mkdir(parents=True, exist_ok=True)
            save(dest/'result.json', value)
            return value, dest/'result.json'
        with patch.object(batch, 'launch_trial', side_effect=fake):
            return batch.dispatch(self.args, self.output, self.rows, self.summary, [])

    def test_compile_failure_skips_only_candidate_and_continues(self):
        self.assertEqual(self.launch(['compile_failed', 'measured', 'measured']), 0)
        self.assertEqual(self.calls, [(203, 'baseline'), (204, 'baseline'), (204, 'block')])
        self.assertTrue(self.summary['completed'])
        self.assertFalse(self.summary['all_passed'])
        batch.export(self.output, self.summary, self.rows)
        with (self.output/'performance.csv').open(encoding='utf-8-sig', newline='') as stream:
            data = list(csv.DictReader(stream))
        self.assertEqual(data[0]['baseline_us'], '')
        self.assertEqual(data[0]['block_status'], 'skipped_baseline_failed')
        self.assertEqual(float(data[1]['speedup']), 2)

    def test_resume_does_not_rerun_terminal_success_or_failure(self):
        self.launch(['measured', 'runtime_fallback', 'accuracy_failed'])
        self.calls.clear()
        self.assertEqual(self.launch([]), 0)
        self.assertEqual(self.calls, [])

    def test_busy_card_pauses_then_resumes_only_unfinished(self):
        self.assertEqual(self.launch(['measured', 'measured', 'device_unavailable']), 2)
        self.assertFalse(self.summary['completed'])
        self.assertEqual(self.summary['state'], 'waiting_for_device')
        self.calls.clear()
        self.assertEqual(self.launch(['measured', 'measured']), 0)
        self.assertEqual(self.calls, [(204, 'baseline'), (204, 'block')])

    def test_rejected_or_unvalidated_timing_is_never_a_speedup(self):
        for status in ('accuracy_unvalidated', 'runtime_fallback', 'timing_rejected_interference'):
            self.assertIsNone(batch.eligible_us({'status': status,
                'timing': {'performance_eligible': True, 'kernel_us': 1}}))
        for value in (0, float('nan'), float('inf'), -1):
            self.assertIsNone(batch.eligible_us({'status':'measured',
                'timing':{'performance_eligible':True,'kernel_us':value}}))

    def test_orphaned_attempt_not_implicitly_repeated(self):
        dest = self.output/'trial'
        (dest/'attempt-001').mkdir(parents=True)
        with self.assertRaisesRegex(RuntimeError, 'Incomplete attempt needs review'):
            batch.launch_trial(self.args, self.rows[0], self.config, 'baseline', dest, [])

    def test_terminal_worker_result_recovers_after_controller_exit(self):
        dest = self.output/'trial'
        attempt = dest/'attempt-001'
        attempt.mkdir(parents=True)
        save(attempt/'result.json', {'status':'compile_failed','completed':True,'timing':None})
        with patch.object(batch.subprocess, 'Popen') as spawn:
            result, _ = batch.launch_trial(self.args, self.rows[0], self.config, 'baseline', dest, [])
        self.assertEqual(result['status'], 'compile_failed')
        spawn.assert_not_called()

    def test_resume_rejects_changed_selection_or_configuration(self):
        config = {'factor': 2}
        saved = {'selection': [203,204], 'run_config': config}
        batch.validate_resume(saved, self.rows, config)
        with self.assertRaisesRegex(ValueError, 'Resume requires identical'):
            batch.validate_resume(saved, self.rows[:1], config)
        with self.assertRaisesRegex(ValueError, 'Resume requires identical'):
            batch.validate_resume(saved, self.rows, {'factor': 4})


@unittest.skipUnless(sys.platform == 'linux', 'Real inherited flock checks require Linux')
class LinuxSupervision(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def contender(self, path):
        code = ('import fcntl,sys; f=open(sys.argv[1],"a"); '
                '\ntry: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)'
                '\nexcept BlockingIOError: sys.exit(3)')
        return subprocess.run([sys.executable, '-I', '-c', code, str(path)], check=False).returncode

    def test_child_keeps_card_lock_after_controller_handle_closes(self):
        args = SimpleNamespace(physical_card=6, task_lock=self.root/'task.lock', card_lock=None)
        path = self.root/'card-6.lock'
        child = None
        try:
            with patch.object(batch, 'lock_root', return_value=self.root), batch.batch_locks(args) as locks:
                self.assertEqual(self.contender(path), 3)
                child = subprocess.Popen([sys.executable, '-I', '-c',
                    'import sys; print("ready",flush=True); sys.stdin.read(1)'],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                    pass_fds=tuple(f.fileno() for f in locks))
                self.assertEqual(child.stdout.readline().strip(), 'ready')
            self.assertEqual(self.contender(path), 3)
            child.communicate('x', timeout=5)
            self.assertEqual(self.contender(path), 0)
        finally:
            if child is not None and child.poll() is None:
                child.kill(); child.communicate()

    def test_timeout_has_terminal_failure_and_no_timing(self):
        args = SimpleNamespace(worker_timeout=1)
        command = [sys.executable, '-I', '-c', 'import time; time.sleep(90)']
        with patch.object(batch, 'worker_command', return_value=command):
            result, path = batch.launch_trial(args, {}, {}, 'baseline', self.root/'trial', [])
        self.assertEqual(result['status'], 'worker_timeout')
        self.assertTrue(result['completed'])
        self.assertIsNone(result['timing'])
        self.assertTrue(path.is_file())


if __name__ == '__main__':
    unittest.main()
