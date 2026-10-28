"""Read operator metadata and build deterministic launch plans without Torch."""
import importlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def operators():
    return read_json(ROOT / 'catalog.json')['operators']


def resolve(name):
    matches = [entry for entry in operators() if name in (entry['id'], entry['name'])]
    if len(matches) != 1:
        raise ValueError(f'Unknown or ambiguous operator: {name}')
    entry = matches[0]
    path = (ROOT / entry['path']).resolve()
    if not path.is_relative_to(ROOT / 'operators'):
        raise ValueError('Operator path must remain inside operators/')
    return entry, path


def module(entry, suffix):
    return importlib.import_module(entry['path'].replace('/', '.') + '.' + suffix)


def plans(name, suite='all', variants='all'):
    if name in ('all', 'graph', 'flaggems', 'liger') or ',' in name:
        return [plan for entry in select(name)
                for plan in plans(entry['id'], suite, variants)]
    entry, path = resolve(name)
    if entry.get('runner') == 'liger':
        if variants == 'manual':
            raise ValueError('Liger batch contains original baselines; no manual variants')
        cases = read_json(path / 'cases.json')['cases']
        dtype = {'fp32':'float32', 'bf16':'bfloat16'}.get(suite)
        return [dict(operator_id=entry['id'], operator=entry['name'], case=case,
                     variant_id='baseline', required_input_file=None,
                     source_kind=entry['implementation'], runner='tools/run.py run')
                for case in cases if suite in ('all','smoke') or case['dtype'] == dtype]
    if entry.get('runner') == 'graph_tests':
        if variants == 'manual':
            raise ValueError('TG benchmarks have explicit baselines; no manual candidate has been measured')
        cases = read_json(path / 'cases.json')['cases']
        dtype = {'fp32':'float32', 'bf16':'bfloat16'}.get(suite)
        return [dict(operator_id=entry['id'], operator=entry['name'], case=case,
                     variant_id='baseline', compile_options={'enable_graph_optimize':False},
                     required_input_file=None, runner='tools/run.py run') for case in cases
                if suite == 'all' or (suite == 'smoke' and case['suite'] == 'smoke') or
                (dtype is not None and case['dtype'] == dtype)]
    baseline = module(entry, 'baseline')
    manual = module(entry, 'manual_tile')
    launch = module(entry, 'launch')
    configurations = {}
    if variants in ('all', 'baseline'):
        configurations[baseline.ID] = baseline.CONFIG
    if variants in ('all', 'manual'):
        configurations.update(manual.CONFIGS)
    cases = read_json(path / 'cases.json')['cases']
    if suite == 'smoke':
        cases = cases[:1]
    return [dict(operator_id=entry['id'], operator=entry['name'], case=case,
                 baseline_id=baseline.ID, variant_id=vid, config=dict(config),
                 grid=list(launch.grid(case['shape'], config)),
                 launch_options={'num_warps': 32, 'num_stages': 2})
            for case in cases if suite in ('all', 'smoke') or case['suite'] == suite
            for vid, config in configurations.items()]


def select(name):
    if name in ('all', 'graph', 'flaggems', 'liger'):
        runners = {'graph':'graph_tests', 'liger':'liger', 'flaggems':None}
        return [entry for entry in operators() if name == 'all' or
                entry.get('runner') == runners[name]]
    result = [resolve(part.strip())[0] for part in name.split(',')]
    if len({entry['id'] for entry in result}) != len(result):
        raise ValueError('Duplicate operator in selection')
    return result
