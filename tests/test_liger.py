import csv
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from opbench.catalog import plans, select
from opbench.liger import verify_suite
from opbench.liger_profile import _trace_samples, measure
from opbench.liger_runner import failure_status


class LigerTests(unittest.TestCase):
    def test_complete_frozen_scope_and_unmodified_functions(self):
        result = verify_suite()
        self.assertEqual(result['operators'],46)
        self.assertEqual(result['historical'],dict(measured=42,compile_failed=3,blocked_by_forward_compile=1))

    def test_batch_selectors_do_not_mix_families(self):
        self.assertEqual([len(select(s)) for s in ('flaggems','graph','liger')],[8,7,46])
        batches = [select(s) for s in ('flaggems','graph','liger','fla')]
        self.assertEqual(len(select('all')), sum(len(batch) for batch in batches))
        self.assertEqual({e['id'] for e in select('all')},
                         {e['id'] for batch in batches for e in batch})
        self.assertEqual(len(plans('liger','all','baseline')),46)
        self.assertEqual(plans('LK46')[0]['source_kind'],'experimental')
        with self.assertRaises(ValueError):
            plans('liger','all','manual')

    def test_dependency_failure_is_not_target_compile_success_or_failure(self):
        self.assertEqual(failure_status('CompilationError: MaxThreads out of range',False),
                         'blocked_by_dependency_compile')
        self.assertEqual(failure_status("CompilationError: 'dtype' object is not callable",True),'compile_failed')
        self.assertEqual(failure_status('AssertionError: tensor mismatch',True),'accuracy_failed')
        self.assertEqual(failure_status('NPU device fault',False),'runtime_failed')

    def test_mspti_path_preserves_telemetry_and_compiled_evidence(self):
        profile = {'variants': {'baseline': {
            'median_us': 2.0, 'samples': [{'duration_us': 2.0}] * 15,
            'rounds': [{'round': r, 'median_us': 2.0} for r in range(3)]}}}
        fake_torch = SimpleNamespace(npu=SimpleNamespace(synchronize=lambda: None))
        with tempfile.TemporaryDirectory() as tmp, \
             patch.dict('sys.modules', {'torch': fake_torch}), \
             patch('opbench.liger_profile._host_pid', return_value=123), \
             patch('opbench.liger_profile._telemetry', return_value={'errors': []}), \
             patch('opbench.liger_profile._compiled_evidence', return_value=[{'hash': 'test'}]), \
             patch('opbench.mspti.measure_npu', return_value=profile) as collector:
            result = measure(lambda: None, 'target', '/source.py', tmp, method='mspti')
        self.assertEqual(result['status'], 'MEASURED_KERNEL')
        self.assertEqual(result['method'], 'mspti')
        self.assertEqual(result['us'], 2.0)
        self.assertEqual(len(result['samples_us']), 15)
        self.assertEqual(result['compiled_evidence'], [{'hash': 'test'}])
        self.assertEqual(collector.call_args.args[0][0][2], 'target')

    def test_mspti_rejects_foreign_device_process(self):
        fake_torch = SimpleNamespace(npu=SimpleNamespace(synchronize=lambda: None))
        with tempfile.TemporaryDirectory() as tmp, \
             patch.dict('sys.modules', {'torch': fake_torch}), \
             patch('opbench.liger_profile._host_pid', return_value=123), \
             patch('opbench.liger_profile._telemetry', return_value={'errors': ['FOREIGN_NPU_PROCESS']}), \
             patch('opbench.mspti.measure_npu') as collector:
            result = measure(lambda: None, 'target', '/source.py', tmp, method='mspti')
        self.assertEqual(result['status'], 'INTERFERED_TIMING_REJECTED')
        collector.assert_not_called()

    def profile(self, folder, unknown=False, missing=False):
        events, rows = [], []
        for i in range(15):
            if missing and i==14:
                continue
            events.extend([dict(ph='X',name=f'ligerbase::target::{i}',pid=1,ts=i*100,dur=50),
                           dict(ph='s',cat='async_npu',id=i,pid=1,ts=i*100+1),
                           dict(ph='f',cat='async_npu',id=i,pid=2,ts=i*1000+10000)])
            rows.append({'Name':'vendor' if unknown and i==0 else 'target',
                         'Start Time(us)':str(i*1000+10000),'Duration(us)':str(i+1)})
        (folder/'trace_view.json').write_text(json.dumps(dict(traceEvents=events)),encoding='utf-8')
        with (folder/'kernel_details.csv').open('w',newline='',encoding='utf-8') as handle:
            writer=csv.DictWriter(handle,fieldnames=['Name','Start Time(us)','Duration(us)'])
            writer.writeheader()
            writer.writerows(rows)

    def test_exact_target_samples_and_median(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            self.profile(path)
            result=_trace_samples(path,'target')
            self.assertEqual(result['us'],8)
            self.assertEqual(len(result['samples_us']),15)

    def test_unknown_kernel_or_missing_sample_rejects_timing(self):
        for options in (dict(unknown=True),dict(missing=True)):
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)
                self.profile(path,**options)
                with self.assertRaises(ValueError):
                    _trace_samples(path,'target')
