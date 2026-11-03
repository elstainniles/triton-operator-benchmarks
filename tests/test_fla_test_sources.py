"""Mutation tests for the generated-fixture FLA source contract."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from opbench.fla_test_sources import verify_suite, audit_python


class FlaTestSources(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.group = self.root / 'operators/flash_linear_attention'
        self.op = self.group / 'example'
        self.variant = self.op / 'tests/vv_example-123'
        self.source = self.variant / 'source/test_case.py'
        self.fixture = self.variant / 'generated/fixture.py'
        self.source.parent.mkdir(parents=True)
        self.fixture.parent.mkdir(parents=True)
        self.source.write_text('"""Docstring."""\nfrom __future__ import annotations\n'
                               'from opbench.fla_fixtures import load_fixture\n'
                               'def kernel(x): return x + 1\n'
                               'def test_case(): return load_fixture("input.pt")\n')
        self.fixture.write_text('def make_fixtures(seed=42): return {"input.pt": {"x": seed}}\n')
        original = self.op / 'upstream.py'
        original.write_text('def kernel(x): return x + 1\n')
        self.meta = dict(id='FLA001', kernel_symbol='kernel', source=dict(
            revision='a'*40, path='fla/example.py', snapshot='upstream.py', sha256=self.sha(original)))
        self.record = dict(id='vv_example-123', case_id='vv_example', operator_id='FLA001',
            kernel='kernel', source_path='fla/example.py', entrypoint='source/test_case.py',
            provider='generated/fixture.py', function='test_case', pythonpath=['source'],
            external_input_required=False, input_origin='seeded_cpu_generation', sides=['baseline'],
            path=str(self.variant.relative_to(self.root)), files=[])
        self.local = dict(schema_version=1, operator_id='FLA001', tests=[dict(
            id='vv_example-123', case_id='vv_example', path='vv_example-123/test.json')])
        self.index = dict(schema_version=1, source_revision='a'*40, summary=dict(
            operator_entries=1, test_variants=1, case_ids=1), tests=[dict(
            operator_id='FLA001', id='vv_example-123', case_id='vv_example',
            path=str((self.variant/'test.json').relative_to(self.root)))])
        self.write(self.root/'catalog.json', {'operators': [dict(
            id='FLA001', runner='fla', path=str(self.op.relative_to(self.root)))]})
        self.rehash()
        self.save()

    def sha(self, path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def write(self, path, data):
        path.write_text(json.dumps(data))

    def rehash(self):
        self.record['files'] = [dict(path=str(p.relative_to(self.variant)),sha256=self.sha(p),role=role)
                               for p,role in [(self.source,'test'),(self.fixture,'fixture_provider')]]

    def save(self):
        self.write(self.op/'operator.json', self.meta)
        self.write(self.op/'tests/manifest.json', self.local)
        self.write(self.variant/'test.json', self.record)
        self.write(self.group/'test-sources.json', self.index)

    def test_valid_generated_fixture_sources(self):
        result = verify_suite(self.root)
        self.assertEqual(result['kernel_body_verified'], 1)
        self.assertEqual(result['files'], 2)

    def test_dynamic_loader_metadata_is_hashed_and_checked(self):
        metadata = self.source.parent / 'source-proof.json'
        self.write(metadata, {'definitions': {'kernel': {'module': 'test_case', 'wrapper': 'test_case'}}})
        self.record['files'].append(dict(path='source/source-proof.json',
                                         sha256=self.sha(metadata), role='dependency'))
        self.save()
        self.assertEqual(verify_suite(self.root)['files'], 3)
        metadata.write_text('{"changed": true}')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            verify_suite(self.root)

    def test_missing_and_changed_sources_rejected(self):
        self.source.write_text(self.source.read_text()+'# changed\n')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):
            verify_suite(self.root)
        self.source.unlink()
        with self.assertRaisesRegex(ValueError,'Missing FLA test source'):
            verify_suite(self.root)

    def test_wrong_kernel_mapping_rejected(self):
        self.record['kernel']='other'
        self.save()
        with self.assertRaisesRegex(ValueError,'kernel/source mapping'):
            verify_suite(self.root)

    def test_index_cannot_hide_test(self):
        self.index['tests']=[]
        self.save()
        with self.assertRaisesRegex(ValueError,'omits an operator test manifest'):
            verify_suite(self.root)

    def test_duplicate_selector_rejected(self):
        self.index['tests']*=2
        self.save()
        with self.assertRaisesRegex(ValueError,'Duplicate FLA test selector'):
            verify_suite(self.root)

    def test_provider_cannot_load_files_even_after_rehash(self):
        self.fixture.write_text('def make_fixtures():\n import torch\n return torch.load("missing.pt")\n')
        self.rehash();self.save()
        with self.assertRaisesRegex(ValueError,'External tensor load'):
            verify_suite(self.root)

    def test_provider_cannot_write_files(self):
        self.fixture.write_text('def make_fixtures():\n open("output.pt", "w")\n return {"x": 1}\n')
        self.rehash();self.save()
        with self.assertRaisesRegex(ValueError,'file I/O'):
            verify_suite(self.root)

    def test_original_future_import_order_remains_legal(self):
        audit_python(self.source)
        self.source.write_text('from opbench.fla_fixtures import load_fixture\n'+self.source.read_text())
        with self.assertRaises(SyntaxError):
            audit_python(self.source)

    def test_counts_must_match_actual_sources(self):
        self.index['summary']['test_variants']=2
        self.save()
        with self.assertRaisesRegex(ValueError,'summary mismatch'):
            verify_suite(self.root)

    def test_original_kernel_body_difference_rejected_after_rehash(self):
        self.source.write_text(self.source.read_text().replace('x + 1','x + 2'))
        self.rehash();self.save()
        with self.assertRaisesRegex(ValueError,'kernel body differs'):
            verify_suite(self.root)


if __name__=='__main__':
    unittest.main()
