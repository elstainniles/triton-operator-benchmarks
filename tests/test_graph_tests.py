import ast
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from opbench.catalog import operators, plans, read_json, resolve
from opbench.graph_tests import launch_calls, verify_graph_suite
from tools.run_graph_tests import validate_request


class GraphTests(unittest.TestCase):
    def test_all_imported_launches_disable_graph_without_changing_algorithm(self):
        entries = [e for e in operators() if e.get('runner') == 'graph_tests']
        self.assertEqual(len(entries), 7)
        total = 0
        for entry in entries:
            _, path = resolve(entry['id'])
            result = verify_graph_suite(path)
            total += result['graph_disabled_launches']
            self.assertEqual(plans(entry['id'])[0]['compile_options'], {'enable_graph_optimize':False})
        self.assertEqual(total, 9)

    def test_graph_true_is_rejected_even_if_file_hash_updated(self):
        _, path = resolve('TG04')
        metadata = copy.deepcopy(read_json(path/'operator.json'))
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            for record in metadata['files']:
                source = (path/record['snapshot']).read_bytes()
                (target/record['snapshot']).parent.mkdir(parents=True, exist_ok=True)
                (target/record['snapshot']).write_bytes(source)
                changed = (path/record['executable']).read_bytes().replace(
                    b'enable_graph_optimize=False', b'enable_graph_optimize=True')
                (target/record['executable']).write_bytes(changed)
                record['executable_sha256'] = hashlib.sha256(changed).hexdigest()
            (target/'operator.json').write_text(json.dumps(metadata), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'enable_graph_optimize=False'):
                verify_graph_suite(target)

    def test_missing_pt_rejected_before_device_initialization(self):
        _, path = resolve('TG01')
        with self.assertRaisesRegex(ValueError, 'Missing upstream input'):
            validate_request(read_json(path/'operator.json'), None, [])

    def test_moe_subset_requires_valid_ids(self):
        _, path = resolve('TG07')
        metadata = read_json(path/'operator.json')
        validate_request(metadata, None, [0,99])
        for ids in ([-1], [100]):
            with self.assertRaises(ValueError):
                validate_request(metadata, None, ids)

    def test_cannot_apply_flaggems_manual_plan_to_imported_suite(self):
        with self.assertRaises(ValueError):
            plans('TG04', variants='manual')


if __name__ == '__main__':
    unittest.main()
