"""Frozen BH=8 / 32x32 CV cases and audited historical batch-matmul results."""
import ast
import hashlib
import math
import statistics

from .catalog import ROOT, read_json
from .environment import digest
from .liger import GROUP, verify_operator
from .liger_cv_large import IDS, load_case

MANIFEST = GROUP / 'cv-small-manifest.json'
LOWERING = 'rank3_full_tile_batch_v2'


def validate_merge(launch, lowering, factor, grid, full_masks):
    if (factor not in (2, 4, 8) or len(grid) != 3 or grid[0] % factor
            or launch.get('runtime_status') != 'APPLIED'
            or launch.get('applied_factor') != factor or launch.get('selected_axis') != 0
            or list(launch.get('original_grid', [])) != list(grid)
            or list(launch.get('batch_grid', [])) != [grid[0] // factor, *grid[1:]]
            or lowering != LOWERING or full_masks is not True):
        raise RuntimeError('Requested full-tile BH batch merge was not applied; reject candidate timing')


def validate_compiled(compiled, factor, grid):
    report = getattr(compiled, 'block_batching', {})
    validate_merge(getattr(compiled, 'last_launch', {}), report.get('cv_lowering'), factor, grid,
                   report.get('guard', {}).get('require_full_masks'))
    artifact = getattr(compiled, 'candidate', None)
    metadata = getattr(artifact, 'metadata', None)
    if (getattr(metadata, 'enable_hivm_batch_matmul', None) is not True
            or getattr(metadata, 'auto_blockify_enabled', None) is not False
            or getattr(metadata, 'default_dot_input_precision', None) != 'ieee'):
        raise RuntimeError('Candidate must enable batch matmul, disable auto blockify, and retain IEEE precision')
    dots = [line.strip() for line in getattr(artifact, 'asm', {}).get('ttir', '').splitlines()
            if 'tt.dot ' in line]
    if len(dots) != 1 or dots[0].count(f'tensor<{factor}x32x32xf32>') != 3:
        raise RuntimeError('Expected one independent rank-3 FP32 dot; reject candidate timing')
    return dict(cv_lowering=LOWERING, require_full_masks=True, dot=dots[0],
                enable_hivm_batch_matmul=True, auto_blockify_enabled=False,
                binary_sha256=hashlib.sha256(artifact.kernel).hexdigest())


def verify_all():
    """No Torch/NPU imports: compare frozen inputs and recompute every timing."""
    records = read_json(MANIFEST)['entries']
    assert [r['operator_id'] for r in records] == list(IDS)
    samples = applied = confirmed = probes = 0
    for record in records:
        entry, path, case = load_case(record['operator_id'], 'small')
        verify_operator(path)
        old, = read_json(path / 'cases.json')['cases']
        assert case['original_case_file_sha256'] == digest(path / 'cases.json')
        assert case['shape'] == old['shape'] == [2, 4, 32, 32]
        assert case['dtype'] == old['dtype'] == 'float32' and case['type'] == 'CV'
        assert case['tensors'] == old['tensors']
        expected = {k: ast.literal_eval(v) for k, v in old['historical_bound_arguments'].items()
                    if k not in old['tensors']}
        assert case['scalar_arguments'] == expected
        assert case['tile'] == {k: 32 for k in ('BLOCK_SIZE_M', 'BLOCK_SIZE_N', 'BLOCK_SIZE_K')}
        assert case['grid'] == [8, 1, 1] and case['seed'] == 20261031
        function, = [n for n in ast.parse((path / 'kernels.py').read_text(encoding='utf-8')).body
                     if isinstance(n, ast.FunctionDef)]
        assert {a.arg for a in function.args.args} == set(expected) | set(case['tensors'])
        for name, spec in case['tensors'].items():
            assert spec['shape'] == ([32, 32] if name == 'mask_ptr' else [2, 4, 32, 32])
            assert spec['stride'] == [math.prod(spec['shape'][i + 1:]) for i in range(len(spec['shape']))]
        measurement = ROOT / record['measurement']
        assert digest(measurement) == record['measurement_sha256']
        history = read_json(measurement)
        assert history['repo_kernel_sha256'] == digest(path / 'kernels.py')
        assert history['kernel'] == entry['name'] and history['shape'] == case['shape']
        assert history['tile'] == case['tile'] and history['original_grid'] == case['grid']
        assert [v['factor'] for v in history['variants']] == [1, 2, 4, 8]
        for variant in history['variants']:
            correct = variant['correctness']
            assert correct['passed'] and correct['inputs_unchanged'] and correct['canaries_ok']
            assert variant['compiler_metadata']['enable_hivm_batch_matmul'] is True
            assert variant['compiler_metadata']['default_dot_input_precision'] == 'ieee'
            if variant['factor'] > 1:
                assert variant['status'] == 'MEASURED_APPLIED'
                validate_merge(variant['runtime_launch'], variant['cv_lowering'], variant['factor'],
                               case['grid'], variant['require_full_masks'])
                proof = variant['lowering_proof']
                assert proof['same_binary']
                assert proof['original_binary_sha256'] == proof['diagnostic_binary_sha256'] == variant['binary_sha256']
                assert any('call @batch_mma' in line for line in proof['batch_mma_lines'])
                assert variant['compiler_metadata']['auto_blockify_enabled'] is False
                applied += 1
            else:
                assert variant['status'] == 'MEASURED_BATCH_CONTROL'
                assert variant['variant'] == 'batch_control' and not variant['confirmed_gain']
            rounds = variant['paired_rounds']
            assert len(rounds) == variant['rounds'] == (4 if variant['replayed'] else 2)
            assert variant['replayed'] == (variant['factor'] > 1 and all(x > 1.05 for x in variant['screen_round_speedups']))
            for pair in rounds:
                for side in ('baseline', 'candidate'):
                    raw = pair[side + '_samples_us']
                    assert len(raw) == 15 and all(math.isfinite(x) and x > 0 for x in raw)
                    assert statistics.median(raw) == pair[side + '_us']
                    samples += len(raw)
                assert pair['speedup'] == pair['baseline_us'] / pair['candidate_us']
            for side in ('baseline', 'candidate'):
                assert variant[side + '_us'] == statistics.median(p[side + '_us'] for p in rounds)
            assert variant['speedup'] == variant['baseline_us'] / variant['candidate_us']
            assert variant['confirmed_gain'] == (len(rounds) == 4 and all(p['speedup'] > 1.05 for p in rounds))
            confirmed += variant['confirmed_gain']
        best = min((v for v in history['variants'] if v['confirmed_gain']), key=lambda v: v['candidate_us'])
        assert all(best[k] == value for k, value in history['best'].items())
        expected_probes = {'tail': 'CV_FULL_TILE_MASK_NOT_PROVEN', 'nondivisible': 'CV_GRID_NOT_DIVISIBLE'}
        assert {p['probe'] for p in history['guard_probes']} == set(expected_probes)
        for probe in history['guard_probes']:
            assert probe['status'] == 'CORRECT_BASELINE_FALLBACK'
            assert probe['reason'] == expected_probes[probe['probe']]
            assert probe['correctness']['passed'] and probe['correctness']['bitwise_equal_baseline']
            probes += 1
    return dict(operators=len(records), applied_candidates=applied, confirmed_gain_candidates=confirmed,
                guard_probes=probes, device_samples=samples)
