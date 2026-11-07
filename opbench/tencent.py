"""Offline plans and source fidelity checks for the user-provided Tencent suite."""
import ast
import hashlib

from .catalog import module, read_json, select, resolve


def plans(entry, path, suite, variants):
    if variants == 'manual':
        raise ValueError('Tencent entries contain original baselines only; no manual variants published')
    baseline, launch = module(entry, 'baseline'), module(entry, 'launch')
    cases = read_json(path / 'cases.json')['cases']
    return [dict(operator_id=entry['id'], operator=entry['name'], case=case,
                 baseline_id=baseline.ID, variant_id=baseline.ID, config=dict(baseline.CONFIG),
                 grid=list(launch.grid(case['shape'], baseline.CONFIG)), launch_options={},
                 required_input_file=None, timing='profiler', runner='tools/run.py run')
            for case in cases if suite in ('all', 'smoke') or case['suite'] == suite]


def verify_suite():
    entries = select('Tencent')
    assert len(entries) == 6
    for entry in entries:
        _, path = resolve(entry['id'])
        meta = read_json(path / 'operator.json')
        source = path / meta['source']['snapshot']
        assert hashlib.sha256(source.read_bytes()).hexdigest() == meta['source']['sha256']
        original = {n.name: n for n in ast.parse(source.read_text(encoding='utf-8')).body
                    if isinstance(n, ast.FunctionDef)}
        actual = {n.name: n for n in ast.parse((path / 'kernels.py').read_text(encoding='utf-8')).body
                  if isinstance(n, ast.FunctionDef)}
        assert set(actual) == {meta['kernel_symbol'], *meta['helpers']}
        for name, function in actual.items():
            assert ast.dump(function) == ast.dump(original[name]), (entry['id'], name)
        assert meta['id'] == entry['id'] and meta['path'] == entry['path']
        assert meta['type'] == entry['type'] == 'vv'
        assert meta['required_input_file'] is None
        assert len(read_json(path / 'cases.json')['cases']) == 1
        assert not module(entry, 'manual_tile').CONFIGS
        planned, = plans(entry, path, 'all', 'baseline')
        assert planned['config'] == meta['configurations']['baseline']
        assert meta['launch_options'] == planned['launch_options'] == {}
    return dict(operators=len(entries), cases=len(entries), source_functions_unchanged=True,
                device_validation='not_revalidated_for_repository_entry_points')
