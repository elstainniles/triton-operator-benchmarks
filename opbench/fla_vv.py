"""Workbook-ordered FLA VV selection; planning never imports Torch or Triton."""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from .catalog import ROOT, read_json
from .fla_test_sources import inside

GROUP = ROOT / 'operators/flash_linear_attention'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def indices(text, available):
    """Inclusive ranges, with deterministic workbook order handled by selected()."""
    if text == 'all':
        return set(available)
    wanted = set()
    for part in text.split(','):
        match = re.fullmatch(r'\s*(\d+)\s*(?:-\s*(\d+)\s*)?', part)
        if not match:
            raise ValueError('Use all, a number, or inclusive ranges such as 1-20,25,30-40')
        first, last = int(match[1]), int(match[2] or match[1])
        if first > last or first not in available or last not in available:
            raise ValueError(f'Range {part!r} is outside {min(available)}..{max(available)}')
        wanted.update(range(first, last + 1))
    if not wanted or not wanted <= set(available):
        raise ValueError('Unknown case in selection')
    return wanted


def selected(cases='all', rows=None, operator='all'):
    records = read_json(GROUP / 'vv-manifest.json')['rows']
    if rows is not None and cases != 'all':
        raise ValueError('Choose --cases (VV ordinal) OR --rows (Excel row), not both')
    key, selector = ('row', rows) if rows is not None else ('index', cases)
    wanted = indices(selector, {r[key] for r in records})
    records = [r for r in records if r[key] in wanted]
    if operator != 'all':
        ids = set(operator.split(','))
        known = {r['operator_id'] for r in read_json(GROUP / 'vv-manifest.json')['rows']}
        if not ids <= known:
            raise ValueError('Unknown VV operator ID: ' + ','.join(sorted(ids - known)))
        records = [r for r in records if r['operator_id'] in ids]
    if not records:
        raise ValueError('Selection contains no VV rows')
    return records


def load_case(row):
    path = inside(ROOT, row['path'] + '/cases-vv.json')
    data = read_json(path)
    matches = [s for s in data['cases'] if s['id'] == row['id']]
    if len(matches) != 1 or data['operator_id'] != row['operator_id']:
        raise ValueError('VV per-operator mapping mismatch')
    case = matches[0]
    if any(case[k] != row[k] for k in ('id', 'row', 'index', 'operator_id', 'kernel', 'path')):
        raise ValueError('VV row identity changed')
    return case


def load_test(case, config):
    info = inside(ROOT, config['test_manifest'])
    if sha(info) != config['test_manifest_sha256']:
        raise ValueError('VV imported test manifest changed: ' + config['test_id'])
    spec = read_json(info)
    expected = ROOT / case['path'] / 'tests' / config['test_id'] / 'test.json'
    if info.resolve() != expected.resolve():
        raise ValueError('VV test is not under its operator')
    if (spec['id'] != config['test_id'] or spec['kernel'] != case['kernel']
            or spec['operator_id'] != case['operator_id']):
        raise ValueError('VV test identity mismatch')
    for key, source in (('config', 'autotune_config'), ('invocation', 'invocation'),
                        ('seed', 'seed'), ('correctness_kind', 'correctness_kind'),
                        ('validation_standard', 'validation_standard')):
        if config[key] != spec[source]:
            raise ValueError('VV configuration differs from imported test: ' + key)
    if not all(v is None or config['config'].get(k) == v for k, v in case['table_config'].items()):
        raise ValueError('VV launch config differs from recorded workbook values')
    selection = config.get('input_selection')
    if selection:
        tree = ast.parse(inside(info.parent, spec['entrypoint']).read_text(encoding='utf-8'))
        definitions = [n.value for n in tree.body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == selection['global_name'] for t in n.targets)]
        if len(definitions) != 1 or not isinstance(definitions[0], ast.List):
            raise ValueError('VV input selection must refer to one original static configuration list')
        item = definitions[0].elts[selection['index']]
        expected = {ast.literal_eval(k): ast.unparse(v) if isinstance(v, ast.Attribute) else ast.literal_eval(v)
                    for k, v in zip(item.keys, item.values) if ast.literal_eval(k) != 'desc'}
        if expected != selection['expected']:
            raise ValueError('VV input selection differs from original configuration')
    return info.parent, spec


def verify_all():
    from .fla_test_runner import verify_files
    data = read_json(GROUP / 'vv-manifest.json')
    rows = data['rows']
    if ([r['index'] for r in rows] != list(range(1, 318))
            or [r['row'] for r in rows] != list(range(203, 520))
            or len({r['id'] for r in rows}) != 317):
        raise ValueError('Frozen VV workbook order changed')
    imported = {r['id']: r for r in read_json(GROUP / 'test-sources.json')['tests']}
    used, kinds = {}, Counter()
    for row in rows:
        case = load_case(row)
        if case['operator_type'] != 'VV' or not case['shape_dtype'] or not case['configurations']:
            raise ValueError('VV row lacks shape/configuration')
        used.setdefault(row['path'], []).append(row['id'])
        if len({c['test_id'] for c in case['configurations']}) != len(case['configurations']):
            raise ValueError('Duplicate configuration in VV row')
        for config in case['configurations']:
            if imported[config['test_id']]['path'] != config['test_manifest']:
                raise ValueError('VV configuration is not in original test index')
            directory, spec = load_test(case, config)
            verify_files(directory, spec)
            kinds[config['correctness_kind']] += 1
    actual = {p.relative_to(ROOT).parent.as_posix() for p in GROUP.rglob('cases-vv.json')}
    if actual != set(used):
        raise ValueError('VV manifest omits a per-operator case file')
    for path, ids in used.items():
        if [c['id'] for c in read_json(ROOT / path / 'cases-vv.json')['cases']] != ids:
            raise ValueError('VV per-operator cases contain an omitted or reordered row')
    counts = dict(rows=len(rows), configurations=sum(kinds.values()), operator_entries=len(used),
                  kernel_symbols=len({r['kernel'] for r in rows}))
    if counts != data['summary']:
        raise ValueError('VV summary differs from the frozen mappings')
    return dict(counts, acceptance_kinds=dict(kinds), device_validation='NOT_RUN')


def plan(args):
    planned = []
    for row in selected(args.cases, args.rows, args.operator):
        case = load_case(row)
        for config in case['configurations']:
            load_test(case, config)
        planned.append(dict(case, external_input_required=False))
    return planned


def main(argv=None):
    parser = argparse.ArgumentParser(description='FLA VV workbook rows: verify, plan, serial run and resume.')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('verify')
    for name in ('plan', 'run', '_worker'):
        p = commands.add_parser(name)
        selection = p.add_mutually_exclusive_group()
        selection.add_argument('--cases', default='all', help='VV ordinals 1..317, e.g. 1-20,25-40 (inclusive)')
        selection.add_argument('--rows', help='Excel rows 203..519, e.g. 203-222 (inclusive)')
        p.add_argument('--operator', default='all', help='Optional FLA ID(s), including FLA_EXTRA_*')
        if name == 'plan':
            continue
        p.add_argument('--physical-card', type=int, choices=range(8), required=True)
        p.add_argument('--environment', type=Path, required=True)
        p.add_argument('--output', type=Path, required=True)
        p.add_argument('--factor', type=int, default=1, help='1: baseline only; >=2: baseline then block candidate')
        p.add_argument('--axis', type=int, choices=(0, 1, 2), default=0)
        p.add_argument('--rounds', type=int, default=2)
        p.add_argument('--samples', type=int, default=15, help='Samples per round')
        p.add_argument('--warmup', type=int, default=5, help='Warmup launches per round')
        p.add_argument('--correctness-only', action='store_true')
        p.add_argument('--task-lock', type=Path)
        p.add_argument('--card-lock', type=Path)
        p.add_argument('--worker-timeout', type=int, default=1800)
        if name == 'run':
            p.add_argument('--resume', action='store_true')
        else:
            p.add_argument('--test-id', required=True)
            p.add_argument('--variant', choices=('baseline', 'block'), required=True)
            p.add_argument('--lock-fds', required=True)
    args = parser.parse_args(argv)
    if args.command in ('run', '_worker') and (min(args.factor, args.rounds, args.samples, args.worker_timeout) < 1 or args.warmup < 0):
        parser.error('factor, rounds, samples and timeout must be positive; warmup must be nonnegative')
    try:
        if args.command == 'verify':
            print(json.dumps(verify_all(), indent=2))
        elif args.command == 'plan':
            print(json.dumps(plan(args), indent=2, ensure_ascii=False))
        else:
            from .fla_vv_runner import run_batch, run_one
            return (run_batch if args.command == 'run' else run_one)(args)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.error(str(error))
    return 0
