"""Source and dispatch contracts run on a host without importing Torch/Triton."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from opbench import cli
from opbench.catalog import ROOT, read_json, select
from opbench.fla import plans, verify_operator, verify_scope
from opbench.fla_runner import run
from tools.verify_repository import HISTORICAL_HARNESS, verify_historical_harness


class FlaTests(unittest.TestCase):
    def fixture(self, path, source='@triton.jit\ndef kernel(x):\n    return x + 1\n'):
        (path / 'upstream').mkdir()
        (path / 'upstream/source.py').write_text(source)
        (path / 'kernels.py').write_text(source)
        digest = hashlib.sha256(source.encode()).hexdigest()
        entry = dict(id='FLA_TEST', name='kernel', path='operators/flash_linear_attention/test',
                     runner='fla', entry_mode='source_archive', type='vv')
        meta = dict(entry, kernel_symbol='kernel', baseline_id='baseline',
                    adaptation_reason='No standalone launch and independent reference are registered.',
                    source=dict(repository='https://github.com/fla-org/flash-linear-attention',
                                revision='a' * 40, path='source.py', snapshot='upstream/source.py',
                                sha256=digest, symbol='kernel', license='MIT'),
                    extraction=dict(path='kernels.py', sha256=digest, ast_preserved=True))
        (path / 'operator.json').write_text(json.dumps(meta))
        (path / 'cases.json').write_text(json.dumps(dict(cases=[], external_input_required=False)))
        return entry, meta

    def save_meta(self, path, meta):
        meta['extraction']['sha256'] = hashlib.sha256((path / 'kernels.py').read_bytes()).hexdigest()
        (path / 'operator.json').write_text(json.dumps(meta))

    def test_archive_plan_and_run_are_explicit_and_device_free(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            entry, meta = self.fixture(path)
            schedule = plans(entry, path, 'all', 'baseline')
            self.assertEqual(schedule[0]['status'], 'not_executable')
            self.assertIsNone(schedule[0]['case'])
            args = SimpleNamespace(operator=entry['id'], output=path / 'result', suite='all',
                                   variants='baseline', case_id=[], environment=None,
                                   physical_card=None, correctness_only=False)
            with patch('opbench.fla_runner.resolve', return_value=(entry, path)), \
                    patch('opbench.fla_runner.plans', return_value=schedule), \
                    patch('opbench.fla_runner.configure') as configure, \
                    patch('opbench.fla_runner.Monitor') as monitor, \
                    patch('opbench.fla_runner.module') as module:
                result = run(args)
            configure.assert_not_called()
            monitor.assert_not_called()
            module.assert_not_called()
            self.assertTrue(result['completed'])
            self.assertFalse(result['all_passed'])
            self.assertFalse(result['device_execution'])
            self.assertIsNone(result['cases'][0]['median_us'])
            self.assertEqual(result['reason'], meta['adaptation_reason'])

    def test_archive_is_not_a_manual_variant(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            entry, _ = self.fixture(path)
            with self.assertRaisesRegex(ValueError, 'no manual'):
                plans(entry, path, 'all', 'manual')

    def test_body_change_rejected_even_with_updated_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            _, meta = self.fixture(path)
            (path / 'kernels.py').write_text((path / 'kernels.py').read_text().replace('+ 1', '+ 2'))
            self.save_meta(path, meta)
            with self.assertRaisesRegex(ValueError, 'changes original'):
                verify_operator(path)

    def test_fixed_autotune_policy_only_removes_original_autotune(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            source = '@triton.autotune(configs=[], key=["x"])\n@triton.jit\ndef kernel(x):\n    return x + 1\n'
            _, meta = self.fixture(path, source)
            (path / 'kernels.py').write_text(source.split('\n', 1)[1])
            self.save_meta(path, meta)
            with self.assertRaises(ValueError):
                verify_operator(path)
            meta['decorator_policy'] = 'fixed_autotune_candidate'
            self.save_meta(path, meta)
            verify_operator(path)
            (path / 'kernels.py').write_text(source)
            self.save_meta(path, meta)
            with self.assertRaisesRegex(ValueError, 'changes original'):
                verify_operator(path)

    def test_jit_decorator_must_be_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            _, meta = self.fixture(path)
            meta['decorator_policy'] = 'fixed_autotune_candidate'
            (path / 'kernels.py').write_text((path / 'kernels.py').read_text().replace('@triton.jit', '@triton.jit(debug=True)'))
            self.save_meta(path, meta)
            with self.assertRaises(ValueError):
                verify_operator(path)

    def test_default_helper_cannot_be_replaced_with_fast_branch(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            _, meta = self.fixture(path)
            helpers = ("if os.environ.get('FLA_USE_FAST_OPS', '0') == '1':\n"
                       '    @triton.jit\n    def exp(x): return fast(x)\nelse:\n'
                       '    @triton.jit\n    def exp(x): return normal(x)\n')
            (path / 'upstream/helper.py').write_text(helpers)
            meta.update(helper_policy='default_FLA_USE_FAST_OPS_0', kernel_helpers=[dict(
                snapshot='upstream/helper.py', symbol='exp',
                sha256=hashlib.sha256(helpers.encode()).hexdigest())])
            original = (path / 'kernels.py').read_text()
            (path / 'kernels.py').write_text(original + '\n@triton.jit\ndef exp(x): return normal(x)\n')
            self.save_meta(path, meta)
            verify_operator(path)
            (path / 'kernels.py').write_text(original + '\n@triton.jit\ndef exp(x): return fast(x)\n')
            self.save_meta(path, meta)
            with self.assertRaisesRegex(ValueError, 'changes helper'):
                verify_operator(path)

    def test_archive_cannot_claim_runnable_cases(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            self.fixture(path)
            (path / 'cases.json').write_text(json.dumps(dict(cases=[dict(id='bad')], external_input_required=False)))
            with self.assertRaisesRegex(ValueError, 'no runnable cases'):
                verify_operator(path)

    def test_source_path_cannot_escape_operator(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            _, meta = self.fixture(path)
            meta['source']['snapshot'] = '../outside.py'
            self.save_meta(path, meta)
            with self.assertRaisesRegex(ValueError, 'inside its operator'):
                verify_operator(path)

    def test_dependency_hash_and_path_are_audited(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            _, meta = self.fixture(path)
            source = 'VERSION = 1\n'
            (path / 'upstream/dependency.py').write_text(source)
            meta['dependencies'] = [dict(path='upstream/dependency.py',
                                         sha256=hashlib.sha256(source.encode()).hexdigest())]
            self.save_meta(path, meta)
            verify_operator(path)
            (path / 'upstream/dependency.py').write_text('VERSION = 2\n')
            with self.assertRaisesRegex(ValueError, 'dependency hash mismatch'):
                verify_operator(path)
            meta['dependencies'][0]['path'] = '../outside.py'
            self.save_meta(path, meta)
            with self.assertRaisesRegex(ValueError, 'inside its operator'):
                verify_operator(path)

    def scope_data(self):
        group = ROOT / 'operators/flash_linear_attention'
        entries = select('fla')
        metadata = [read_json(ROOT / entry['path'] / 'operator.json') for entry in entries]
        return [entries, metadata, read_json(group / 'source-manifest.json'),
                read_json(group / 'scope.json'), read_json(group / 'import-manifest.json')['inventory']]

    def test_catalog_cannot_drop_an_imported_operator(self):
        data = self.scope_data()
        verify_scope(*data)
        data[0] = data[0][1:]
        with self.assertRaisesRegex(ValueError, 'full source manifest'):
            verify_scope(*data)
        data[0] = []
        with self.assertRaisesRegex(ValueError, 'full source manifest'):
            verify_scope(*data)

    def test_source_symbol_identity_cannot_be_duplicated(self):
        data = self.scope_data()
        data[1][1]['source'] = copy.deepcopy(data[1][0]['source'])
        data[1][1]['kernel_symbol'] = data[1][0]['kernel_symbol']
        with self.assertRaisesRegex(ValueError, 'Duplicate FLA source path and symbol'):
            verify_scope(*data)

    def test_generated_plan_filters_cases_and_checks_catalog(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            entry, meta = self.fixture(path)
            entry['entry_mode'] = meta['entry_mode'] = 'generated_benchmark'
            (path / 'benchmark.py').write_text('def make_case(config): pass\n')
            cases = [dict(id='fp32', shape=[17], dtype='float32', suite='smoke', seed=42),
                     dict(id='bf16', shape=[17], dtype='bfloat16', suite='all', seed=42)]
            (path / 'cases.json').write_text(json.dumps(dict(cases=cases, external_input_required=False)))
            self.save_meta(path, meta)
            self.assertEqual(len(plans(entry, path, 'smoke', 'baseline')), 1)
            self.assertEqual(plans(entry, path, 'bf16', 'baseline')[0]['case']['id'], 'bf16')
            entry['entry_mode'] = 'source_archive'
            with self.assertRaisesRegex(ValueError, 'catalog and metadata disagree'):
                plans(entry, path, 'all', 'baseline')

    def test_list_supports_explicit_fla_selector(self):
        with patch.object(sys, 'argv', ['run.py', 'list', '--operator', 'fla']), \
                patch('opbench.cli.select', return_value=[dict(id='FLA_TEST')]) as select, \
                contextlib.redirect_stdout(io.StringIO()) as output:
            cli.main()
        select.assert_called_once_with('fla')
        self.assertEqual(json.loads(output.getvalue()), [dict(id='FLA_TEST')])

    def test_host_imports_do_not_load_accelerators(self):
        code = ('import sys;sys.path.insert(0, ' + repr(str(ROOT)) + ');'
                'import opbench.fla,opbench.fla_runner;'
                'assert "torch" not in sys.modules and "triton" not in sys.modules')
        subprocess.run([sys.executable, '-I', '-c', code], check=True)

    def test_historical_harness_is_fixed_not_an_arbitrary_hash_override(self):
        for name, expected in HISTORICAL_HARNESS.items():
            verify_historical_harness(name, expected)
            with self.assertRaisesRegex(AssertionError, 'Unknown historical harness'):
                verify_historical_harness(name, '0' * 64)


if __name__ == '__main__':
    unittest.main()
