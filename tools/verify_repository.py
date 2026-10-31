"""Offline integrity audit; intentionally imports neither Torch nor Triton."""
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opbench.catalog import module, operators, plans, read_json, resolve
from opbench.graph_tests import verify_graph_suite
from opbench.liger import verify_suite as verify_liger_suite
from opbench.fla import verify_suite as verify_fla_suite


# These files changed since the Liger measurements were collected. Historical evidence
# continues to bind their exact original bytes, never the newly modified code.
HISTORICAL_HARNESS = {
    'catalog.py': 'f6f72b8071a91acda3e7d24d4457b817383edf1f911dde211c02167da9fe268e',
    'graph_runner.py': 'e6625eafb80229057b9e93b206cb7962bba4a3bb466403d992fa7242acb7e206',
    'runner.py': 'a71e1bdab749604c8cd6d1aa9b82ecc6f73260c2093cbef8ad1638610e339d33',
    'liger_profile.py': '71f8885f2a1fb74fbb3bbd43f464a6cbdbbd31f0703d890e272ad7040535a6ac',
    'liger_runner.py': 'a8154dc4fac624cec436ca49a12b1986a66a622215ce0d7491556be24b778e6b',
    'cli.py': 'cf012d361d0e7f9b2f80670090a73b3cd543424b1e1e683217541a695af781d7',
    'batch.py': 'acbce294a176735f9931fceafc145e1735ae54742449af620df6e40637146cdc',
}


def verify_historical_harness(filename, expected):
    if filename in HISTORICAL_HARNESS:
        assert expected == HISTORICAL_HARNESS[filename], ('Unknown historical harness', filename)
        path = ROOT / 'docs/historical-harness-20261028' / filename
        manifest = read_json(path.parent / 'manifest.json')
        assert manifest['source_revision'] == '20cc2f0ad0efafbd494341bff65cbaff0d89d69e'
        assert manifest['files'] == HISTORICAL_HARNESS
    else:
        path = ROOT / 'opbench' / filename
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, ('Historical harness changed', filename)


def same(a, b):
    return math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10)


def verify():
    all_entries = operators()
    assert len({o['id'] for o in all_entries}) == len(all_entries)
    assert len({o['path'] for o in all_entries}) == len(all_entries)
    entries = [e for e in all_entries if not e.get('runner')]
    liger = verify_liger_suite()
    fla = verify_fla_suite()
    validation = read_json(ROOT/'docs/liger-validation-20261028.json')
    assert validation['total'] == len(validation['evidence']) == 25
    assert validation['counts'] == {'measured':25}
    assert len({row['operator_id'] for row in validation['evidence']}) == 25
    changed_harness = set()
    for row in validation['evidence']:
        evidence = read_json(ROOT/row['path'])
        _, path = resolve(row['operator_id'])
        assert evidence['status'] == row['status'] == 'measured'
        assert evidence['correctness']['passed'] and evidence['all_observed_occupancy_checks_passed']
        assert len(evidence['samples_us']) == 15
        assert same(statistics.median(evidence['samples_us']), row['median_us'])
        for filename, digest in evidence['source_hashes'].items():
            assert hashlib.sha256((path/filename).read_bytes()).hexdigest() == digest
        for filename, digest in evidence['harness_sha256'].items():
            verify_historical_harness(filename, digest)
            if hashlib.sha256((ROOT/'opbench'/filename).read_bytes()).hexdigest() != digest:
                changed_harness.add(filename)
        assert hashlib.sha256((path/'cases.json').read_bytes()).hexdigest() == evidence['cases_sha256']
        assert read_json(path/'operator.json')['current_runner_validation']['status'] == 'SECOND_BATCH_CASE_VALIDATED'
    graph_entries = [e for e in all_entries if e.get('runner') == 'graph_tests']
    graph_files = graph_launches = 0
    for entry in graph_entries:
        _, path = resolve(entry['id'])
        counts = verify_graph_suite(path)
        graph_files += counts['files']
        graph_launches += counts['graph_disabled_launches']
        assert plans(entry['id'])[0]['compile_options'] == {'enable_graph_optimize': False}
        cases = read_json(path/'cases.json')
        metadata = read_json(path/'operator.json')
        assert cases['external_input_required'] is False and metadata['required_input_file'] is None
        assert metadata['input_mode'] == 'generated_benchmark' and (path/'benchmark.py').is_file()
        assert len(cases['cases']) == metadata['case_count']
        assert len({c['id'] for c in cases['cases']}) == len(cases['cases'])
        assert plans(entry['id'], 'smoke', 'baseline')
    group = ROOT / 'operators/triton_graph_optimization'
    manifest = read_json(group / 'source-manifest.json')
    assert manifest['local_test_suites'] == len(graph_entries)
    assert len(manifest['files']) == graph_files
    for record in manifest['files']:
        path = ROOT / record['path']
        assert record in [dict(operator_id=read_json(path/'operator.json')['id'],
                              path=record['path'], **f)
                          for f in read_json(path/'operator.json')['files']]
    reference = manifest['reference_document']
    assert hashlib.sha256((group/reference['snapshot']).read_bytes()).hexdigest() == reference['sha256']
    counters = Counter()
    fp32 = []
    environment = read_json(ROOT / 'docs/historical-environment.json')
    for ent in entries:
        _, path = resolve(ent['id'])
        meta = read_json(path / 'operator.json')
        baseline, manual, launch = (module(ent, name) for name in ('baseline', 'manual_tile', 'launch'))
        assert meta['baseline_id'] == baseline.ID
        configurations = {baseline.ID: baseline.CONFIG, **manual.CONFIGS}
        assert baseline.ID not in manual.CONFIGS
        assert configurations == meta['configurations']
        source = path / meta['source']['snapshot']
        assert hashlib.sha256(source.read_bytes()).hexdigest() == meta['source']['sha256']
        def function(file, name):
            return next(n for n in ast.parse(file.read_text(encoding='utf-8')).body
                        if isinstance(n, ast.FunctionDef) and n.name == name)
        original = function(source, meta['source']['symbol'])
        original.decorator_list = [d for d in original.decorator_list if ast.unparse(d).startswith('triton.jit')]
        extracted = function(path / 'kernels.py', meta['source']['symbol'])
        assert ast.dump(original) == ast.dump(extracted), ('Changed source function', ent['id'])
        measurements = read_json(path / 'results/20261026/measurements.json')['measurements']
        comparisons = read_json(path / 'results/20261026/comparison.json')['comparisons']
        by_id = {r['measurement_id']:r for r in measurements}
        assert len(by_id) == len(measurements)
        paired = set()
        for row in measurements:
            assert row['operator_id'] == ent['id']
            assert row['source_sha256'] == meta['source']['sha256']
            assert row['compiler_revision'] == environment['triton_ascend_revision']
            assert row['libtriton_sha256'] == environment['libtriton_sha256']
            assert row['grid'] == list(launch.grid(row['shape'], configurations[row['variant_id']]))
            assert row['config'] == dict(configurations[row['variant_id']], **meta['launch_options'])
            assert (path / 'results/20261026' / row['environment']).resolve().is_file()
            counters[row['status']] += 1
            if row['status'] == 'passed':
                assert row['correctness']['passed'] and row['correctness']['guards']
                assert row['timing_scope'] == 'NPU kernel duration' and not row['timing_exclusion']
                rounds, repeats = (2, 3) if row['smoke'] else (3, 5)
                assert len(row['samples_ms']) == rounds * repeats
                assert len(row['profile_rounds']) == rounds
                # Exported samples preserve the historical round/repeat order.
                for i, round_result in enumerate(row['profile_rounds']):
                    actual = statistics.median(row['samples_ms'][i*repeats:(i+1)*repeats]) * 1000
                    assert same(actual, round_result['sum_median_us'])
                assert same(row['median_ms'] * 1000,
                            statistics.median(r['sum_median_us'] for r in row['profile_rounds']))
            else:
                assert row['median_ms'] is None and not row['samples_ms'] and row['failure']
        case_ids = {c['id'] for c in read_json(path / 'cases.json')['cases']}
        assert case_ids == {c['case_id'] for c in comparisons if not c['smoke']}
        for compare in comparisons:
            group = 'generalization-20261026/' + compare['run'] + '/' + compare['case_id']
            variants = {v['id']:v for v in compare['variants']}
            assert set(variants) == set(configurations)
            assert compare['timing_eligible'] and not compare['timing_exclusion']
            for vid, variant in variants.items():
                mid = group + '/' + vid
                row = by_id[mid]
                paired.add(mid)
                assert row['comparison_group'] == group
                assert (row['shape'],row['dtype'],row['smoke']) == (compare['shape'],compare['dtype'],compare['smoke'])
                assert row['status'] == variant['status']
                assert row['config'] == variant['config']
                if row['median_ms'] is not None:
                    assert same(row['median_ms'] * 1000, variant['median_us'])
                    assert same(100 * (1-variant['median_us']/compare['baseline_us']), variant['time_reduction_pct'])
            assert same(variants[baseline.ID]['median_us'], compare['baseline_us'])
            best = variants[compare['best_id']]
            assert best['status'] == 'passed' and same(best['median_us'], compare['best_us'])
            assert same(compare['best_us'], min(v['median_us'] for v in variants.values() if v['median_us'] is not None))
            assert same(compare['speedup'], compare['baseline_us']/compare['best_us'])
            assert same(compare['time_reduction_pct'], 100*(1-compare['best_us']/compare['baseline_us']))
            counters['comparisons'] += 1
            counters['smoke' if compare['smoke'] else 'bf16' if compare['dtype']=='torch.bfloat16' else 'fp32'] += 1
            if not compare['smoke'] and compare['dtype'] == 'torch.float32':
                fp32.append(compare)
        assert paired == set(by_id)
        assert len(plans(ent['id'])) == len(case_ids) * len(configurations)
    summary = read_json(ROOT / 'docs/historical-summary.json')
    for name in ('comparisons', 'smoke', 'fp32', 'bf16', 'passed', 'failed'):
        assert counters[name] == summary['counts'][name], (name, counters[name])
    assert len(entries) == summary['counts']['operators']
    assert sum(counters[k] for k in ('passed', 'failed')) == summary['counts']['candidates']
    for name, rows in [('fp32', fp32), ('original_kernel_fp32', [r for r in fp32 if r['operator_id'] not in {'P07','P08','P09'}])]:
        expected = summary[name]
        assert len(rows) == expected['configurations']
        assert sum(r['time_reduction_pct']>5 for r in rows) == expected['improved_over_5pct']
        assert same(math.exp(statistics.mean(math.log(r['speedup']) for r in rows)), expected['geomean_speedup'])
        assert same(statistics.mean(r['speedup'] for r in rows), expected['mean_speedup'])
        assert same(statistics.mean(r['time_reduction_pct'] for r in rows), expected['mean_time_reduction_pct'])
    smoke = read_json(ROOT / 'docs/standalone-smoke-20261028.json')
    expected_smoke = {(p['operator_id'],p['case']['id']):p
                      for e in all_entries if e.get('runner') in (None, 'graph_tests')
                      for p in plans(e['id'],'smoke','baseline')}
    measured = {(r['operator_id'],r['case']['id']):r for r in smoke['measurements']}
    assert len(measured) == len(smoke['measurements']) == 18 and measured.keys() == expected_smoke.keys()
    assert smoke['counts'] == dict(operator_entries=15, smoke_configurations=18, correct=18, measured=18)
    for key, row in measured.items():
        planned = expected_smoke[key]
        _, path = resolve(key[0])
        assert row['status'] == 'measured' and row['correctness']['passed'] and row['correctness']['guards']
        assert row['case'] == planned['case']
        assert row['config'] == planned.get('config', planned['case'].get('tile'))
        for filename, digest in row['source_hashes'].items():
            assert hashlib.sha256((path/filename).read_bytes()).hexdigest() == digest, (key,filename)
        samples = row['timing']['samples']
        assert {(s['round'],s['repeat']) for s in samples} == {(i,j) for i in range(3) for j in range(5)}
        assert len(samples) == 15 and all(s['duration_us'] > 0 for s in samples)
        medians = [statistics.median(s['duration_us'] for s in samples if s['round'] == i) for i in range(3)]
        assert same(row['median_us'], statistics.median(medians))
        assert same(row['timing']['median_us'], row['median_us'])
        if key[0].startswith('TG'):
            assert row['graph_optimization'] is False and row['launch_options']['enable_graph_optimize'] is False
        if key[0] in ('TG04','TG05','TG06'):
            assert row['launch_options']['force_simt_only'] is True
        metadata = read_json(path/'operator.json')['current_runner_validation']
        assert metadata['status'] == 'SMOKE_VALIDATED' and row['case']['id'] in metadata['cases']
        assert metadata['full_matrix_validated'] is False
    for file in ROOT.rglob('*'):
        if not file.is_file() or any(part in {'.git','__pycache__','runs','.venv'} for part in file.relative_to(ROOT).parts):
            continue
        if file.suffix == '.py':
            ast.parse(file.read_text(encoding='utf-8'), filename=str(file))
        if file.suffix == '.md':
            text = file.read_text(encoding='utf-8')
            text = re.sub(r'^```[^\n]*\n.*?^```[^\n]*$', '', text, flags=re.M | re.S)
            for link in re.findall(r'\[[^\]]*\]\(([^)]+)\)', text):
                if '://' in link or link.startswith('#'):
                    continue
                assert (file.parent / link.split('#')[0]).exists(), (file, link)
    assert 'torch' not in sys.modules and 'triton' not in sys.modules
    return dict(status='PASS', entries=len(all_entries), flaggems_entries=len(entries),
                imported_graph_suites=len(graph_entries), imported_graph_files=graph_files,
                graph_disabled_launches=graph_launches, verified_baseline_smoke=len(measured), liger=liger,
                verified_liger_entry_cases=validation['total'], fla=fla,
                historical_liger_harness_changed=sorted(changed_harness), **counters)


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
