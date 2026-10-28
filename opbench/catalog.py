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
    entry, path = resolve(name)
    baseline = module(entry, 'baseline')
    manual = module(entry, 'manual_tile')
    launch = module(entry, 'launch')
    configurations = {}
    if variants in ('all', 'baseline'):
        configurations[baseline.ID] = baseline.CONFIG
    if variants in ('all', 'manual'):
        configurations.update(manual.CONFIGS)
    cases = read_json(path / 'cases.json')['cases']
    return [dict(operator_id=entry['id'], operator=entry['name'], case=case,
                 baseline_id=baseline.ID, variant_id=vid, config=dict(config),
                 grid=list(launch.grid(case['shape'], config)),
                 launch_options={'num_warps': 32, 'num_stages': 2})
            for case in cases if suite == 'all' or case['suite'] == suite
            for vid, config in configurations.items()]
