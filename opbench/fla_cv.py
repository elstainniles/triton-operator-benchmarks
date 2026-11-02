"""Portable selection and provenance checks for the frozen 201 FLA CV rows."""
import argparse
import ast
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

from .catalog import ROOT, read_json, resolve
from .fla import local_file

GROUP = ROOT / 'operators/flash_linear_attention'
SEEDS = (20261102, 20261103)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selected(cases='all', operator='all'):
    rows = read_json(GROUP / 'cv-manifest.json')['rows']
    if cases == 'tested':
        rows = [r for r in rows if r['historical_status'] not in ('CONTROL_FLOW_BLOCKED', 'SOURCE_NOT_FOUND')]
    elif cases != 'all':
        wanted = {int(v.removeprefix('cv-row-')) for v in cases.split(',')}
        available = {r['row'] for r in rows}
        if not wanted or not wanted <= available:
            raise ValueError('Unknown workbook row; use 2..202 or cv-row-NNN')
        rows = [r for r in rows if r['row'] in wanted]
    if operator != 'all':
        ids = {resolve(v.strip())[0]['id'] for v in operator.split(',')}
        rows = [r for r in rows if r['operator_id'] in ids]
    if not rows:
        raise ValueError('Selection contains no frozen CV cases')
    return rows


def load_case(row):
    if row.get('operator_id') is None:
        return None, None
    entry, path = resolve(row['operator_id'])
    if str(path.relative_to(ROOT)).replace('\\', '/') != row['path']:
        raise ValueError('CV manifest/catalog path mismatch')
    cases = read_json(path / 'cases-cv.json')['cases']
    matches = [s for s in cases if s['id'] == row['id']]
    if len(matches) != 1:
        raise ValueError('Expected one per-operator CV case')
    s = matches[0]
    if s['symbol'] != row['kernel'] or s['operator_id'] != entry['id'] or s['row'] != row['row']:
        raise ValueError('CV case identity mismatch')
    return path, s


def body_hash(node):
    body = ast.Module(body=node.body, type_ignores=[])
    try:
        serialized = ast.dump(body, include_attributes=False, show_empty=True)
    except TypeError:
        serialized = ast.dump(body, include_attributes=False)
    return hashlib.sha256(serialized.encode()).hexdigest()


def verify_case(path, s):
    meta = read_json(path / 'operator.json')
    if s['source_sha256'] != meta['source']['sha256'] or s['source_path'] != meta['source']['path']:
        raise ValueError('CV source identity changed')
    module = local_file(path, s['module'])
    if sha(module) != s['module_sha256']:
        raise ValueError('CV executable source hash mismatch')
    tree = ast.parse(module.read_bytes())
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    for body in s['bodies']:
        if body['name'] not in functions or body_hash(functions[body['name']]) != body['body_sha256']:
            raise ValueError('CV JIT function body changed: ' + body['name'])
        source = ast.parse(local_file(path, body['source']).read_bytes())
        matches = [n for n in ast.walk(source) if isinstance(n, ast.FunctionDef)
                   and n.name == body['name'] and body_hash(n) == body['body_sha256']]
        if not matches:
            raise ValueError('CV function body not found in its original snapshot')
        if not any(ast.dump(n.args) == ast.dump(functions[body['name']].args) for n in matches):
            raise ValueError('CV function signature changed')
    if s['seeds'] != list(SEEDS) or not s['tensors']:
        raise ValueError('CV input seeds/tensors missing')
    if set(s['args']) != {a.arg for a in functions[s['symbol']].args.args}:
        raise ValueError('CV launch arguments do not match original signature')
    for arg in s['args'].values():
        if isinstance(arg, dict) and (set(arg) != {'tensor'} or arg['tensor'] not in s['tensors']):
            raise ValueError('Unknown CV tensor binding')
    for info in s['tensors'].values():
        if not info['shape'] or any(type(n) is not int or n <= 0 for n in info['shape']):
            raise ValueError('CV shapes must be explicit positive extents')
    if s['reference_exp'] not in ('interpreter_default', 'fp64_then_declared_dtype'):
        raise ValueError('Unknown reference exp policy')
    return dict(source_sha256=s['source_sha256'], module_sha256=s['module_sha256'])


def verify_all():
    rows = selected()
    if [r['row'] for r in rows] != list(range(2, 203)) or len({r['id'] for r in rows}) != 201:
        raise ValueError('Frozen workbook row accounting changed')
    count = Counter(r['historical_status'] for r in rows)
    for row in rows:
        path, spec = load_case(row)
        if spec is not None:
            verify_case(path, spec)
            history = read_json(path / 'results/20261102-cv' / (row['id'] + '.json'))
            if history['status'] != row['historical_status'] or history['row'] != row['row']:
                raise ValueError('Historical CV row mismatch')
            if history['candidate_us'] is not None:
                raise ValueError('This imported batch had no applied merge')
            if history['status'] == 'MEASURED':
                import statistics
                samples = history['samples']
                if len(samples) != 2 or any(len(x) != 15 or any(v <= 0 for v in x) for x in samples):
                    raise ValueError('Incomplete historical CV samples')
                if abs(sum(statistics.median(x) for x in samples) / 2 - history['baseline_us']) > 1e-9:
                    raise ValueError('Historical CV statistic mismatch')
            elif history['baseline_us'] is not None:
                raise ValueError('A failed/unexecuted case cannot have a baseline')
    return dict(rows=len(rows), operators=len({r['operator_id'] for r in rows if r['operator_id']}),
                source_backed_rows=sum(r['operator_id'] is not None for r in rows), historical_counts=dict(count))


def load_kernel(path, s):
    verify_case(path, s)
    name = '_opbench_fla_cv_' + s['operator_id']
    spec = importlib.util.spec_from_file_location(name, path / s['module'])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def plan(args):
    result = []
    for row in selected(args.cases, args.operator):
        path, spec = load_case(row)
        status = 'source_not_found' if spec is None else (
            'control_flow_not_revalidated' if row['historical_status'] == 'CONTROL_FLOW_BLOCKED' else 'ready')
        result.append(dict(row=row['row'], id=row['id'], operator_id=row['operator_id'], kernel=row['kernel'],
                           status=status, historical_status=row['historical_status'],
                           tensors=spec['tensors'] if spec else None, grid=spec.get('grid') if spec else None,
                           config=spec['config'] if spec else None, external_input_required=False))
    return result


def main():
    parser = argparse.ArgumentParser(description='Frozen FLA CV cases; no external .pt or source checkout.')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('verify')
    p = commands.add_parser('plan')
    p.add_argument('--cases', default='all', help='all, tested, or comma-separated workbook rows')
    p.add_argument('--operator', default='all', help='all or existing FLA ID(s)')
    for name in ('run', '_worker'):
        p = commands.add_parser(name)
        p.add_argument('--cases', default='all')
        p.add_argument('--operator', default='all')
        p.add_argument('--factor', type=int, default=2)
        p.add_argument('--axis', choices=('all','0','1','2'), default='all')
        p.add_argument('--physical-card', type=int, choices=range(8), required=True)
        p.add_argument('--environment', type=Path, required=True)
        p.add_argument('--output', type=Path, required=True)
        p.add_argument('--task-lock', type=Path)
        p.add_argument('--card-lock', type=Path)
        p.add_argument('--include-control-flow', action='store_true', help='Attempt historically blocked, unvalidated adapters explicitly')
        p.add_argument('--correctness-only', action='store_true')
        p.add_argument('--reference-timeout', type=int, default=600)
        p.add_argument('--worker-timeout', type=int, default=1800)
        if name == 'run':
            p.add_argument('--resume', action='store_true')
    p = commands.add_parser('_reference')
    p.add_argument('--cases', required=True)
    p.add_argument('--seed', type=int, choices=SEEDS, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'verify':
        print(json.dumps(verify_all(), indent=2))
    elif args.command == 'plan':
        print(json.dumps(plan(args), indent=2))
    elif args.command == '_reference':
        from .fla_cv_reference import reference_worker
        reference_worker(args)
    else:
        if args.factor < 1 or args.reference_timeout < 1 or args.worker_timeout < 1:
            parser.error('Factor and timeouts must be positive')
        from .fla_cv_runner import run_batch, run_one
        (run_batch if args.command == 'run' else run_one)(args)
