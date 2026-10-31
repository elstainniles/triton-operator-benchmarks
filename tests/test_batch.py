import csv
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from opbench.batch import run
from opbench.catalog import read_json
from opbench.runner import save


class BatchTests(unittest.TestCase):
    def execute(self, folder, statuses, exit_code=0):
        args = SimpleNamespace(operator='TG04,TG06', case_id=[], suite='smoke', variants='baseline',
                               output=folder/'run', environment=folder/'environment.json',
                               physical_card=6, correctness_only=False)
        calls = []

        def child(cmd, **kwargs):
            destination = Path(cmd[cmd.index('--output')+1])
            destination.mkdir()
            status = statuses[len(calls)]
            calls.append(destination.name)
            save(destination/'result.json', dict(completed=exit_code == 0, cases=[dict(
                case=dict(id='smoke', shape=[1,2,32], dtype='float32'),
                status=status, median_us=2.0 if status == 'measured' else None)]))
            return SimpleNamespace(returncode=exit_code)

        with patch('opbench.batch.subprocess.run', side_effect=child):
            if exit_code:
                with self.assertRaisesRegex(RuntimeError, 'stopped'):
                    run(args)
            else:
                run(args)
        return calls, read_json(args.output/'summary.json'), args.output

    def test_completed_compile_failure_is_not_reported_as_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            calls, summary, output = self.execute(Path(directory), ['compile_failed','measured'])
            self.assertEqual(calls, ['TG04','TG06'])
            self.assertTrue(summary['completed'])
            self.assertFalse(summary['all_passed'])
            self.assertEqual(summary['counts'], {'compile_failed':1,'measured':1})
            self.assertFalse(summary['operators'][0]['all_passed'])
            with (output/'performance.csv').open(newline='') as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]['kernel_us'], '')
            self.assertEqual(float(rows[1]['kernel_us']), 2.0)

    def test_runtime_fault_stops_dispatch_and_preserves_partial_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            calls, summary, _ = self.execute(Path(directory), ['runtime_failed'], exit_code=1)
            self.assertEqual(calls, ['TG04'])
            self.assertFalse(summary['completed'])
            self.assertFalse(summary['all_passed'])
            self.assertEqual(summary['counts'], {'runtime_failed':1})

    def test_mspti_is_forwarded_to_batch_children(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            args = SimpleNamespace(operator='TG04', case_id=[], suite='smoke', variants='baseline',
                                   output=folder/'run', environment=folder/'env.json',
                                   physical_card=6, correctness_only=False, timing='mspti')
            def child(cmd, **kwargs):
                self.assertEqual(cmd[cmd.index('--timing') + 1], 'mspti')
                self.assertEqual(cmd[cmd.index('--physical-card') + 1], '6')
                self.assertEqual(cmd[cmd.index('--environment') + 1], str(args.environment.resolve()))
                destination = Path(cmd[cmd.index('--output') + 1])
                destination.mkdir()
                save(destination/'result.json', dict(completed=True, cases=[dict(
                    case=dict(id='smoke', shape=[1], dtype='float32'), status='measured', median_us=1)]))
                return SimpleNamespace(returncode=0)
            with patch('opbench.batch.subprocess.run', side_effect=child):
                run(args)

    def test_source_archives_do_not_stop_batch_or_require_a_device(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            entries = [dict(id='FLA_ARCHIVE_' + str(i), name='archive_' + str(i), type='vv')
                       for i in range(2)]
            args = SimpleNamespace(operator='fla', case_id=[], suite='all', variants='baseline',
                                   output=folder/'run', environment=None, physical_card=None,
                                   correctness_only=False)

            def child(cmd, **kwargs):
                self.assertNotIn('--physical-card', cmd)
                self.assertNotIn('--environment', cmd)
                destination = Path(cmd[cmd.index('--output')+1])
                destination.mkdir()
                save(destination/'result.json', dict(completed=True, all_passed=False, cases=[dict(
                    case=None, status='not_executable', median_us=None, reason='No standalone adapter')]))
                return SimpleNamespace(returncode=0)

            with patch('opbench.batch.select', return_value=entries), \
                    patch('opbench.batch.plans', return_value=[dict(status='not_executable')]), \
                    patch('opbench.batch.subprocess.run', side_effect=child):
                run(args)
            summary = read_json(args.output/'summary.json')
            self.assertTrue(summary['completed'])
            self.assertFalse(summary['all_passed'])
            self.assertEqual(summary['counts'], {'not_executable':2})


if __name__ == '__main__':
    unittest.main()
