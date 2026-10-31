"""Six frozen large CV cases; direct kernels, CPU references, no package installs.

The input/reference construction follows the completed 2026-10-31 probe.
Historical paired timings are imported separately, never assigned to a new run.
"""
import argparse
import ast
from contextlib import ExitStack
import csv
from datetime import datetime, timezone
import importlib
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import traceback

from .catalog import ROOT, read_json, resolve
from .environment import digest
from .liger import GROUP, verify_operator
from .runner import save

IDS = tuple(f'LK{i:02}' for i in range(7, 13))
MANIFEST = GROUP / 'cv-large-manifest.json'


def load_case(operator, suite='large'):
    if suite not in ('large', 'small'):
        raise ValueError('Unknown CV suite: ' + suite)
    entry, path = resolve(operator)
    if entry['id'] not in IDS:
        raise ValueError('The CV suite contains only LK07-LK12')
    records = read_json(GROUP / f'cv-{suite}-manifest.json')['entries']
    record, = [r for r in records if r['operator_id'] == entry['id']]
    case_path = ROOT / record['case']
    if case_path.resolve() != (path / f'cases-cv-{suite}.json').resolve():
        raise ValueError('CV case path does not match operator')
    if digest(case_path) != record['case_sha256']:
        raise ValueError('CV case differs from frozen manifest')
    case, = read_json(case_path)['cases']
    return entry, path, case


def selected(selector, suite='large'):
    names = IDS if selector == 'all' else selector.split(',')
    rows = [load_case(n.strip(), suite) for n in names]
    if not rows or len({r[0]['id'] for r in rows}) != len(rows):
        raise ValueError('Empty or duplicate operator selection')
    return rows


def verify_all():
    """Validate shapes, source signatures and all imported paired samples offline."""
    records = read_json(MANIFEST)['entries']
    assert [r['operator_id'] for r in records] == list(IDS)
    sample_count = 0
    for record in records:
        entry, path, case = load_case(record['operator_id'])
        verify_operator(path)
        assert case['shape'] == [8, 16, 128, 64] and case['dtype'] == 'float32'
        bound = case['scalar_arguments']
        assert [bound[k] for k in ('batch_size', 'num_heads', 'seq_len', 'head_dim')] == case['shape']
        assert case['dimensions'] == dict(batch_size=8, num_heads=16, seq_len=128, head_dim=64, hidden_size=1024)
        assert case['tile'] == {k: bound[k] for k in ('BLOCK_SIZE_M', 'BLOCK_SIZE_N', 'BLOCK_SIZE_K')}
        function, = [n for n in ast.parse((path / 'kernels.py').read_text(encoding='utf-8')).body
                     if isinstance(n, ast.FunctionDef)]
        assert {a.arg for a in function.args.args} == set(bound) | set(case['tensors'])
        for spec in case['tensors'].values():
            assert spec['dtype'] == 'torch.float32'
            assert spec['stride'] == [math.prod(spec['shape'][i + 1:]) for i in range(len(spec['shape']))]
        output_shape = case['tensors'][case['output']]['shape']
        assert case['grid'] == [128, math.ceil(output_shape[-2] / bound['BLOCK_SIZE_M']),
                               math.ceil(output_shape[-1] / bound['BLOCK_SIZE_N'])]
        assert bound['BLOCK_SIZE_K'] == (64 if entry['id'] in ('LK07', 'LK12') else 128)
        historical_path = ROOT / record['measurement']
        assert digest(historical_path) == record['measurement_sha256']
        history = read_json(historical_path)
        assert history['shape'] == case['shape'] and history['tile'] == case['tile']
        assert history['repo_kernel_sha256'] == digest(path / 'kernels.py')
        assert [v['factor'] for v in history['variants']] == [2, 4, 8]
        for variant in history['variants']:
            assert variant['status'] == 'MEASURED_APPLIED'
            validate_merge(variant['runtime_launch'], variant['cv_lowering'], variant['factor'], case['grid'])
            assert variant['correctness']['bitwise_equal_baseline']
            assert variant['correctness']['inputs_unchanged'] and variant['correctness']['canaries_ok']
            assert len(variant['rounds']) == 2
            for pair in variant['rounds']:
                for side in ('baseline', 'candidate'):
                    samples = pair[side + '_samples_us']
                    assert len(samples) == 15 and all(x > 0 for x in samples)
                    assert statistics.median(samples) == pair[side + '_us']
                    sample_count += len(samples)
                assert pair['speedup'] == pair['baseline_us'] / pair['candidate_us']
            for side in ('baseline', 'candidate'):
                assert variant[side + '_us'] == statistics.median(p[side + '_us'] for p in variant['rounds'])
        fastest = min(history['variants'], key=lambda v: v['candidate_us'])
        assert all(history['best'][k] == fastest[k] for k in history['best'])
    return dict(operators=len(records), applied_candidates=18, device_samples=sample_count)


def validate_merge(launch, lowering, factor, grid):
    expected = [grid[0] // factor, *grid[1:]]
    if (grid[0] % factor or launch.get('runtime_status') != 'APPLIED'
            or launch.get('applied_factor') != factor or launch.get('selected_axis') != 0
            or list(launch.get('original_grid', [])) != list(grid)
            or list(launch.get('batch_grid', [])) != expected
            or lowering != 'static_masked_lanes_v1'):
        raise RuntimeError('Requested first-version CV merge was not applied; reject candidate timing')


def launch_options(factor, suite='large', batch_control=False):
    if suite not in ('large', 'small') or factor not in (1, 2, 4, 8):
        raise ValueError('Unknown CV suite or factor')
    if batch_control and (suite != 'small' or factor != 1):
        raise ValueError('Batch control requires the small suite and factor=1')
    options = dict(enable_hivm_batch_matmul=suite == 'small' and (factor > 1 or batch_control), debug=False)
    if factor > 1:
        options.update(block_batching_cv=True, block_batching_factor=factor, block_batching_axis=0)
    return options


def validate_candidate(compiled, factor, grid, suite):
    if suite == 'small':
        from .liger_cv_small import validate_compiled
        return validate_compiled(compiled, factor, grid)
    else:
        validate_merge(getattr(compiled, 'last_launch', {}),
                       getattr(compiled, 'block_batching', {}).get('cv_lowering'), factor, grid)


def fixture(torch, entry, case):
    """CPU FP64 equations; contiguous FP32 operands; guarded NPU output."""
    B, H, S, D = case['shape']
    torch.manual_seed(case['seed'])
    q = torch.randn(B, H, S, D)
    k, v, dy = torch.randn_like(q), torch.randn_like(q), torch.randn_like(q)
    scale = 1 / math.sqrt(D)
    mask = ((torch.arange(S)[:, None] - torch.arange(S)[None, :]).abs() <= 3).float()
    score = ((q.double() @ k.double().transpose(-1, -2)) * scale).masked_fill(mask == 0, -float('inf'))
    attn = torch.softmax(score, -1).float()
    da = (dy.double() @ v.double().transpose(-1, -2)).float()
    ds = (attn.double() * (da.double() - (attn.double() * da.double()).sum(-1, keepdim=True))).float()
    references = {
        'LK07': lambda: score,
        'LK08': lambda: attn.double() @ v.double(),
        'LK09': lambda: (ds.double() @ k.double()) * scale,
        'LK10': lambda: (ds.double().transpose(-1, -2) @ q.double()) * scale,
        'LK11': lambda: attn.double().transpose(-1, -2) @ dy.double(),
        'LK12': lambda: dy.double() @ v.double().transpose(-1, -2),
    }
    expected = references[entry['id']]().float()
    tensors = dict(Q_ptr=q, K_ptr=k, V_ptr=v, mask_ptr=mask, Attn_ptr=attn,
                   grad_output_ptr=dy, grad_attn_ptr=ds)
    cpu = {n: tensors[n] for n in case['tensors'] if n != case['output']}
    shape = case['tensors'][case['output']]['shape']
    guard = case['correctness']['sentinel_elements_per_side']
    sentinel = case['correctness']['sentinel_value']
    backing = torch.full((math.prod(shape) + 2 * guard,), sentinel, device='npu', dtype=torch.float32)
    output = backing[guard:-guard].view(shape)
    values = {n: tensor.npu() for n, tensor in cpu.items()}
    values[case['output']] = output
    for name, spec in case['tensors'].items():
        assert list(values[name].shape) == spec['shape']
        assert list(values[name].stride()) == spec['stride']
        assert str(values[name].dtype) == spec['dtype']
    values.update(case['scalar_arguments'])

    def check(baseline=None):
        torch.npu.synchronize()
        actual = output.cpu()
        tolerance = {k: case['correctness'][k] for k in ('rtol', 'atol')}
        torch.testing.assert_close(actual, expected, **tolerance)
        assert (backing[:guard] == sentinel).all() and (backing[-guard:] == sentinel).all(), 'Output guard changed'
        assert all(torch.equal(values[n].cpu(), original) for n, original in cpu.items()), 'Input changed'
        info = dict(passed=True, canaries_ok=True, inputs_unchanged=True,
                    max_abs_error=float((actual[torch.isfinite(expected)].double()
                                         - expected[torch.isfinite(expected)].double()).abs().max()))
        if baseline is not None:
            torch.testing.assert_close(actual, baseline, **tolerance)
            info['bitwise_equal_baseline'] = bool(torch.equal(actual, baseline))
        return actual, info

    return values, output, check


def run_one(args):
    import fcntl
    from .device import Monitor, identity
    from .environment import audit, configure
    from .liger import audit_dependency, load_kernel
    from .liger_profile import measure

    entry, path, case = load_case(args.operator, args.suite)
    meta, _ = verify_operator(path)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = dict(origin=f'new_{args.suite}_case_runner', operator=entry, case=case, factor=args.factor,
                  batch_control=args.batch_control,
                  completed=False, status='RUNNING', median_us=None,
                  source_sha256=digest(path / 'kernels.py'), case_sha256=digest(path / f'cases-cv-{args.suite}.json'),
                  harness_sha256={p.name: digest(p) for p in Path(__file__).parent.glob('*.py')},
                  started_utc=datetime.now(timezone.utc).isoformat())
    try:
        expected = read_json(args.environment)
        configure(expected, output, args.physical_card)
        if any(k.startswith('LIGER_KERNEL_IMPL') for k in os.environ):
            raise RuntimeError('Remove inherited Liger implementation overrides')
        os.environ['TRITON_KERNEL_DUMP'] = '1'
        for key, folder in [('TRITON_HOME', 'triton-home'), ('XDG_CACHE_HOME', 'xdg'),
                            ('TORCH_EXTENSIONS_DIR', 'torch-extensions')]:
            location = output / 'cache' / folder
            location.mkdir()
            os.environ[key] = str(location)
        lockroot = Path('/tmp') / f'triton-operator-benchmarks-{os.getuid()}'
        lockroot.mkdir(mode=0o700, exist_ok=True)
        if lockroot.is_symlink() or lockroot.stat().st_uid != os.getuid():
            raise RuntimeError('Invalid card lock directory')
        with ExitStack() as stack:
            # Optional site-wide task lock supplements the shared repository card lock.
            lock_paths = ([args.task_lock] if args.task_lock else [])
            lock_paths += [args.card_lock] if args.card_lock else []
            lock_paths += [lockroot / f'card-{args.physical_card}.lock']
            if len({Path(p).resolve() for p in lock_paths}) != len(lock_paths):
                raise ValueError('Task and card lock paths must be distinct')
            for lock_path in lock_paths:
                lock = stack.enter_context(Path(lock_path).open('a'))
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            owner = identity()
            result['identity'] = owner
            with Monitor(args.physical_card, owner, output / 'device.jsonl') as monitor:
                result['environment'] = audit(expected)
                cann_python = Path(expected['cann_root']) / 'python/site-packages'
                if not (cann_python / 'tbe/common/platform/__init__.py').is_file():
                    raise RuntimeError('Selected CANN is missing tbe platform helpers')
                sys.path.insert(0, str(cann_python))
                result['liger_dependency'] = audit_dependency()
                import torch
                torch.set_num_threads(8)
                if torch.npu.device_count() != 1:
                    raise RuntimeError('Expected one visible NPU')
                torch.npu.set_device(0)
                binding = load_kernel(path, meta)
                result['binding'] = binding
                fn = getattr(importlib.import_module(binding['module']), entry['name'])
                monitor.require_clear()
                start = len(monitor.samples) - 1
                values, target, check = fixture(torch, entry, case)
                grid = tuple(case['grid'])
                base_options = launch_options(1, args.suite)
                target.fill_(float('nan'))
                fn[grid](**values, **base_options)
                baseline, result['baseline_correctness'] = check()
                options = launch_options(args.factor, args.suite, args.batch_control)

                def launch():
                    compiled = fn[grid](**values, **options)
                    if args.factor > 1:
                        validate_candidate(compiled, args.factor, grid, args.suite)
                    return compiled

                target.fill_(float('nan'))
                compiled = launch()
                _, result['correctness'] = check(baseline)
                result.update(options=options, grid=grid, compiler_hash=compiled.hash,
                              last_launch=getattr(compiled, 'last_launch', None))
                if args.factor > 1:
                    result['merge_validation'] = validate_candidate(compiled, args.factor, grid, args.suite)
                if args.correctness_only:
                    result['status'] = 'correct'
                else:
                    result['timing'] = measure(launch, entry['name'], path / 'kernels.py', output)
                    if result['timing']['status'] != 'MEASURED_KERNEL':
                        raise RuntimeError('Timing rejected: interference or missing target evidence')
                    result.update(status='measured', median_us=result['timing']['us'])
                monitor.sample()
                if not monitor.eligible_since(start):
                    result.update(status='timing_rejected_interference', median_us=None)
                    raise RuntimeError('Foreign occupancy during execution; reject timing')
                monitor.require_clear()
        result['completed'] = True
    except BaseException:
        result.update(status='FAILED_REVIEW_REQUIRED', median_us=None, error=traceback.format_exc())
        raise
    finally:
        result['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save(output / 'result.json', result)


def run_batch(args):
    rows = selected(args.operator, args.suite)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary = dict(completed=False, cases=[], suite=args.suite, factor=args.factor, batch_control=args.batch_control)
    try:
        for entry, _, case in rows:
            folder = output / entry['id']
            cmd = [sys.executable, '-I', str(ROOT / f'tools/run_liger_cv_{args.suite}.py'), '_worker',
                   '--operator', entry['id'], '--factor', str(args.factor), '--physical-card', str(args.physical_card),
                   '--environment', str(args.environment.resolve()), '--output', str(folder)]
            if args.correctness_only:
                cmd.append('--correctness-only')
            if args.batch_control:
                cmd.append('--batch-control')
            if args.task_lock:
                cmd.extend(['--task-lock', str(args.task_lock.resolve())])
            if args.card_lock:
                cmd.extend(['--card-lock', str(args.card_lock.resolve())])
            with (output / (entry['id'] + '.log')).open('w', encoding='utf-8') as log:
                process = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
            data = read_json(folder / 'result.json') if (folder / 'result.json').exists() else {}
            summary['cases'].append(dict(operator=entry['name'], case=case['id'], factor=args.factor,
                                         status=data.get('status', 'NO_RESULT'), kernel_us=data.get('median_us'),
                                         exit_code=process.returncode))
            save(output / 'summary.json', summary)
            print(entry['id'], summary['cases'][-1]['status'], flush=True)
            if process.returncode:
                raise RuntimeError('Review ' + entry['id'] + ' log before resuming unfinished cases')
        summary['completed'] = True
    finally:
        save(output / 'summary.json', summary)
        with (output / 'performance.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=['operator', 'case', 'factor', 'status', 'kernel_us', 'exit_code'])
            writer.writeheader()
            writer.writerows(summary['cases'])


def main(suite='large'):
    parser = argparse.ArgumentParser(description=f'Six frozen {suite} Liger CV cases, LK07-LK12.')
    parser.set_defaults(suite=suite, batch_control=False)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('verify')
    plan = commands.add_parser('plan')
    plan.add_argument('--operator', default='all')
    for command in ('run', '_worker'):
        sub = commands.add_parser(command)
        sub.add_argument('--operator', default='all')
        sub.add_argument('--factor', type=int, choices=[1, 2, 4, 8], default=1)
        if suite == 'small':
            sub.add_argument('--batch-control', action='store_true',
                             help='Factor=1 with lower-level batch matmul enabled; no CV merge')
        sub.add_argument('--physical-card', type=int, choices=range(8), required=True)
        sub.add_argument('--environment', type=Path, required=True)
        sub.add_argument('--output', type=Path, required=True)
        sub.add_argument('--task-lock', type=Path, help='Optional existing site-wide validation task lock')
        sub.add_argument('--card-lock', type=Path, help='Optional existing site-wide physical-card lock')
        sub.add_argument('--correctness-only', action='store_true')
    args = parser.parse_args()
    if args.command in ('run', '_worker'):
        launch_options(args.factor, args.suite, args.batch_control)
    if args.command == 'verify':
        import json
        if suite == 'small':
            from .liger_cv_small import verify_all as verify_small
            print(json.dumps(verify_small(), indent=2))
        else:
            print(json.dumps(verify_all(), indent=2))
    elif args.command == 'plan':
        import json
        print(json.dumps([dict(id=e['id'], kernel=e['name'], case=c)
                          for e, _, c in selected(args.operator, suite)], indent=2))
    elif args.command == '_worker':
        run_one(args)
    else:
        run_batch(args)
