"""Fresh in-memory FLA payloads, with no tensor file reads."""
import builtins
import unittest
from unittest.mock import patch
from opbench.fla_fixtures import load_fixture, lookup, use_fixtures
try:
    import torch
except ImportError:
    torch = None


class FlaFixtures(unittest.TestCase):
    def test_exact_suffix_and_unique_basename(self):
        data = {'part/input.pt': 1, 'other/reference.pt': 2, 'input.pt': 3}
        self.assertEqual(lookup(data, 'input.pt'), 3)
        self.assertEqual(lookup(data, '/legacy/part/input.pt'), 1)
        self.assertEqual(lookup(data, '/legacy/reference.pt'), 2)

    def test_missing_and_ambiguous_names_not_guessed(self):
        data = {'left/input.pt': 1, 'right/input.pt': 2}
        for name in ('input.pt', '/elsewhere/input.pt', 'absent.pt'):
            with self.subTest(name=name), self.assertRaises(KeyError):
                lookup(data, name)

    def test_fresh_payload_without_file_reads(self):
        payload = {'x': [1, 2]}
        with use_fixtures({'input.pt': payload}), patch.object(builtins, 'open', side_effect=AssertionError('file read')):
            first = load_fixture('/nonexistent/input.pt', map_location='cpu', weights_only=True)
            first['x'][0] = 3
            self.assertEqual(load_fixture('input.pt'), payload)
        self.assertEqual(payload['x'], [1, 2])

    def test_nested_context_restored_after_exception(self):
        with self.assertRaises(RuntimeError):
            load_fixture('input.pt')
        with use_fixtures({'input.pt': 1}):
            try:
                with use_fixtures({'input.pt': 2}):
                    self.assertEqual(load_fixture('input.pt'), 2)
                    raise ValueError('restore outer')
            except ValueError:
                pass
            self.assertEqual(load_fixture('input.pt'), 1)
        with self.assertRaises(RuntimeError):
            load_fixture('input.pt')

    def test_dictionary_required(self):
        for invalid in (None, []):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                with use_fixtures(invalid):
                    pass

    def test_inline_generated_case_can_have_no_external_payloads(self):
        with use_fixtures({}):
            with self.assertRaises(KeyError):
                load_fixture('missing.pt')

    def test_positional_cpu_map_location(self):
        with use_fixtures({'input.pt': {'x': 42}}):
            self.assertEqual(load_fixture('input.pt', 'cpu'), {'x': 42})
            with self.assertRaises(TypeError):
                load_fixture('input.pt', 'cpu', map_location='cpu')

    @unittest.skipIf(torch is None, 'CPU alias/stride check requires PyTorch')
    def test_aliases_strides_storage_offsets_survive_copy(self):
        base = torch.arange(24, dtype=torch.float32)
        view, other = base[2:20:3], base[3:21:3]
        fixtures = {'input.pt': {'x': view, 'again': view, 'other': other}}
        with use_fixtures(fixtures), patch('torch.load', side_effect=AssertionError('tensor file read')):
            a = load_fixture('input.pt', map_location='cpu')
            b = load_fixture('input.pt', map_location='cpu')
        self.assertIs(a['x'], a['again'])
        self.assertEqual(a['x'].stride(), view.stride())
        self.assertEqual(a['x'].storage_offset(), view.storage_offset())
        self.assertEqual(a['x'].untyped_storage().data_ptr(), a['other'].untyped_storage().data_ptr())
        self.assertNotEqual(a['x'].untyped_storage().data_ptr(), base.untyped_storage().data_ptr())
        self.assertNotEqual(a['x'].untyped_storage().data_ptr(), b['x'].untyped_storage().data_ptr())
        torch.testing.assert_close(a['x'], view, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
