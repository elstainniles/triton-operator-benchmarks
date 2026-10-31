import argparse
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest


REPOSITORY = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('fla_importer_test_subject', REPOSITORY / 'tools/import_fla_kernels.py')
IMPORTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(IMPORTER)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


class ImporterTests(unittest.TestCase):
    def fixture(self, root):
        source, bundle, destination = root / 'source', root / 'bundle', root / 'repository'
        source_file = source / 'fla/ops/example/kernel.py'
        source_file.parent.mkdir(parents=True)
        text = 'import triton\nimport triton.language as tl\n@triton.jit\ndef sample_fwd_kernel(X):\n    value = tl.load(X)\n    tl.store(X, value)\n'
        source_file.write_text(text)
        (source / 'LICENSE').write_text('MIT test fixture\n')
        digest = hashlib.sha256(text.encode()).hexdigest()
        function = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef))
        candidate = {'key': f'fla/ops/example/kernel.py:{function.lineno}:sample_fwd_kernel',
            'source': 'fla/ops/example/kernel.py', 'line': function.lineno,
            'kernel': 'sample_fwd_kernel', 'definition_origin': 'fla_source',
            'original_source_sha256': digest, 'associated_tests': ['vv_example'],
            'static_launch_references': ['fla/ops/example/kernel.py:8']}
        write_json(bundle / 'inventory/kernel_inventory.json', {'source_commit': 'a' * 40, 'kernel_definitions': [candidate]})
        write_json(bundle / 'cases.json', [{'id': 'vv_example', 'group': 'vv', 'original_nodeid': 'tests/test_example.py::test_example'}])
        module = {'module': 'example_module', 'source': candidate['source'], 'source_sha256': digest,
                  'extracted_file': 'example_module.py', 'dependencies': []}
        write_json(bundle / 'extracted/manifest.json', {'source_commit': 'a' * 40,
                    'cases': [{'case_id': 'vv_example', 'modules': [module]}]})
        (bundle / 'extracted/cases/vv_example').mkdir(parents=True)
        (bundle / 'extracted/cases/vv_example/example_module.py').write_text(text)
        write_json(destination / 'catalog.json', {'schema_version': 1, 'operators': []})
        return argparse.Namespace(source_root=source, bundle=bundle, destination=destination,
                                  source_repository='https://example.com/public-source.git',
                                  replace_archives=False, import_manifest=None)

    def run_import(self, args):
        with contextlib.redirect_stdout(io.StringIO()):
            IMPORTER.build(args)

    def test_source_only_rebuild_preserves_generated_adapter_and_catalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            self.run_import(args)
            operator = args.destination / 'operators/flash_linear_attention/sample_fwd_kernel'
            metadata = json.loads((operator / 'operator.json').read_text())
            metadata.update(entry_mode='generated_benchmark', adaptation_status='adapted')
            write_json(operator / 'operator.json', metadata)
            write_json(operator / 'cases.json', {'cases': [{'id': 'reviewed-input', 'seed': 17}]})
            (operator / 'benchmark.py').write_text('# reviewed adapter; must not be overwritten\n')
            catalog = json.loads((args.destination / 'catalog.json').read_text())
            catalog['operators'][0].update(entry_mode='generated_benchmark', reviewed_custom_field='keep')
            write_json(args.destination / 'catalog.json', catalog)
            before = {str(p.relative_to(operator)): p.read_bytes() for p in operator.rglob('*') if p.is_file()}
            args.bundle = None
            args.replace_archives = True
            self.run_import(args)
            after = {str(p.relative_to(operator)): p.read_bytes() for p in operator.rglob('*') if p.is_file()}
            self.assertEqual(before, after)
            self.assertEqual(json.loads((args.destination / 'catalog.json').read_text()), catalog)

    def test_rebuild_rejects_changed_original_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            self.run_import(args)
            args.bundle = None
            args.replace_archives = True
            path = args.source_root / 'fla/ops/example/kernel.py'
            path.write_text(path.read_text().replace('value = tl.load(X)', 'value = tl.load(X) + 1'))
            with self.assertRaisesRegex(ValueError, 'Original source differs'):
                self.run_import(args)


if __name__ == '__main__':
    unittest.main()
