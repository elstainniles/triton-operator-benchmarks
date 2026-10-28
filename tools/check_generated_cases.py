"""CPU input/reference construction and launch-binding checks (no NPU execution).

Requires Torch and Triton. Checks input contracts, not kernel numerical correctness;
use tools/run.py on the NPU for correctness and performance.
"""
import argparse
import ast
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opbench.catalog import module, plans, read_json, resolve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=['smoke','all'], default='smoke')
    parser.add_argument('--operator', default='graph')
    args = parser.parse_args()
    import torch
    torch.set_num_threads(4)
    for plan in plans(args.operator, args.suite, 'baseline'):
        entry, path = resolve(plan['operator_id'])
        metadata = read_json(path/'operator.json')
        case = module(entry, 'benchmark').make_case(plan['case'], device='cpu')
        definitions = [n for f in metadata['files'] for n in ast.parse((path/f['snapshot']).read_bytes()).body
                       if isinstance(n, ast.FunctionDef) and n.name == metadata['kernel_symbol']]
        names = {a.arg for a in definitions[0].args.args}
        assert names == set(case.kwargs), (entry['id'], names-set(case.kwargs), set(case.kwargs)-names)
        assert all(t.device.type == 'cpu' for t in case.cpu_inputs.values())
        if entry['id'] == 'TG07':
            routed = case.cpu_inputs['sorted_token_ids_ptr']
            m, _, _, _, top = plan['case']['shape']
            assert int(case.cpu_inputs['num_tokens_post_padded_ptr'][0]) == routed.numel()
            assert sorted(routed[routed < m*top].tolist()) == list(range(m*top))
            assert case.cpu_inputs['a_ptr'].shape[0] == m
        print(json.dumps(dict(operator=entry['id'], case=plan['case']['id'],
                              input_and_binding_status='PASS')), flush=True)


if __name__ == '__main__':
    main()
