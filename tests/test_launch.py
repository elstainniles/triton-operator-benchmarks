"""Check actual binding against source signatures without importing accelerators."""
import ast
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from opbench.catalog import module, operators, read_json, resolve


class Kernel:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def invoke(*args, **options):
            self.calls.append((grid, args, options))
        return invoke


class LaunchTests(unittest.TestCase):
    def test_tail_shapes_and_source_parameter_binding(self):
        for entry in operators():
            if entry.get('runner') == 'graph_tests':
                continue
            with self.subTest(operator=entry['name']):
                _, path = resolve(entry['id'])
                metadata = read_json(path/'operator.json')
                matrix = entry['id'] in {'P06','P10','P11'}
                shape = (17,513) if matrix else (8193,)
                case = SimpleNamespace(inputs=dict(shape=shape, x='x', y='y', a='a', b='b',
                                                   mask='mask', ab='ab', grad='grad'),
                                       actual=['output'], dtype='torch.float32')
                fake_kernel = Kernel()
                fake = SimpleNamespace(**{metadata['kernel_symbol']:fake_kernel})
                if metadata['implementation'] == 'continuous_adapter':
                    setattr(fake, metadata['source']['symbol'], metadata['source']['symbol'])
                fake_name = entry['path'].replace('/','.') + '.kernels'
                source = ast.parse((path/'kernels.py').read_text(encoding='utf-8'))
                signature = next(n for n in source.body if isinstance(n,ast.FunctionDef)
                                 and n.name == metadata['kernel_symbol']).args.args
                with patch.dict(sys.modules, {fake_name:fake}):
                    for config in metadata['configurations'].values():
                        module(entry,'launch').launch(case, config)
                        grid, args, options = fake_kernel.calls[-1]
                        self.assertEqual(len(args), len(signature))
                        self.assertEqual(options, {'num_warps':32,'num_stages':2})
                        bound = dict(zip((a.arg for a in signature), args))
                        if matrix and entry['id'] != 'P06':
                            self.assertEqual(bound['M'],17)
                            self.assertEqual(bound['H'],513)
                            self.assertEqual(bound['stride_in_m'],1026)
                            self.assertGreaterEqual(grid[0]*config['BM'],17)
                            self.assertLess((grid[0]-1)*config['BM'],17)
                            self.assertGreaterEqual(grid[1]*config['BH'],513)
                            self.assertLess((grid[1]-1)*config['BH'],513)
                        else:
                            n=8721 if matrix else 8193
                            self.assertIn(n,args)
                            self.assertGreaterEqual(grid[0]*config['B'],n)
                            self.assertLess((grid[0]-1)*config['B'],n)


if __name__ == '__main__':
    unittest.main()
