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


def same(a, b):
    return math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10)


def verify():
    entries = operators()
    assert len({o['id'] for o in entries}) == len(entries)
    assert len({o['path'] for o in entries}) == len(entries)
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
    for file in ROOT.rglob('*'):
        if not file.is_file() or any(part in {'.git','__pycache__','runs','.venv'} for part in file.relative_to(ROOT).parts):
            continue
        if file.suffix == '.py':
            ast.parse(file.read_text(encoding='utf-8'), filename=str(file))
        if file.suffix == '.md':
            text = file.read_text(encoding='utf-8')
            for link in re.findall(r'\[[^\]]*\]\(([^)]+)\)', text):
                if '://' in link or link.startswith('#'):
                    continue
                assert (file.parent / link.split('#')[0]).exists(), (file, link)
    assert 'torch' not in sys.modules and 'triton' not in sys.modules
    return dict(status='PASS', entries=len(entries), **counters)


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
