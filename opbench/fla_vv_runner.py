"""Serial, resumable VV batches over the existing self-contained FLA tests."""
from collections import Counter
from contextlib import ExitStack, contextmanager
import csv
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import statistics
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
from unittest.mock import patch

from .catalog import ROOT, read_json
from .fla_vv import GROUP, load_case, load_test, selected, sha
from .runner import compile_error, save

RETRYABLE = {'device_unavailable', 'interrupted', 'infrastructure_error'}
GOOD = {'measured', 'correct'}


def select_input_config(module, selection):
    if not selection:
        return
    choices = getattr(module, selection['global_name'])
    chosen = choices[selection['index']]
    expected = {k: str(v) if k == 'dtype' else v for k, v in chosen.items() if k != 'desc'}
    if expected != selection['expected']:
        raise ValueError('Original input configuration changed before execution')
    setattr(module, selection['global_name'], [chosen])


def target_launch_policy(jit, values, kwargs, config):
    """Fix this target's launch only, for both direct JIT and autotuned wrappers.

    The original grid (including historical invalid grids) is retained. A tile
    passed positionally is replaced in place, never passed a second time.
    """
    kwargs = {k: v for k, v in kwargs.items() if not k.startswith('block_batching_')}
    values = list(values)
    options = dict(config)
    options.update(options.pop('kwargs', {}))
    for name, value in options.items():
        if value is None:
            raise ValueError('Unresolved launch option: ' + name)
        if name in jit.arg_names and jit.arg_names.index(name) < len(values):
            values[jit.arg_names.index(name)] = value
            kwargs.pop(name, None)
        else:
            kwargs[name] = value
    return tuple(values), kwargs


def measure_rounds(measure, capture, output, samples, warmup, rounds):
    readings = []
    for number in range(rounds):
        folder = output / f'round-{number + 1:02d}'
        folder.mkdir()
        reading = measure(capture, folder, samples, warmup)
        reading['median_us'] = statistics.median(reading['samples_us'])
        # Paths in each round are relative to its own directory.
        reading['path'] = folder.relative_to(output).as_posix()
        readings.append(reading)
    values = [v for r in readings for v in r['samples_us']]
    return dict(metric='target_kernel_device_us', statistic='mean_of_round_medians',
                kernel_us=statistics.mean(r['median_us'] for r in readings),
                mean_us=statistics.mean(values), samples_us=values,
                rounds=readings, round_count=rounds, samples_per_round=samples, warmup=warmup)


def run_one(args):
    """Internal worker, holding the controller's inherited task/card locks."""
    import fcntl
    fds = [int(v) for v in args.lock_fds.split(',')]
    if not fds:
        raise ValueError('Worker requires inherited locks from run')
    for fd in fds:
        os.fstat(fd)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    rows = selected(args.cases, args.rows, args.operator)
    if len(rows) != 1:
        raise ValueError('Worker requires exactly one VV row')
    case = load_case(rows[0])
    config = next(c for c in case['configurations'] if c['test_id'] == args.test_id)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = dict(schema_version=1, completed=False, row=case['row'], index=case['index'],
                  case=case['id'], test_id=args.test_id, variant=args.variant,
                  operator_id=case['operator_id'], kernel=case['kernel'], status='starting',
                  timing=None, execution_attempted=False, physical_card=args.physical_card,
                  source_manifest_sha256=config['test_manifest_sha256'],
                  requested_block_factor=args.factor if args.variant == 'block' else None,
                  requested_axis=args.axis if args.variant == 'block' else None,
                  started_epoch=time.time())
    save(output / 'result.json', result)
    phase, cwd = 'setup', Path.cwd()
    try:
        from .environment import configure, audit
        from .device import Monitor, identity
        from . import fla_test_runner as imported
        directory, spec = load_test(case, config)
        result['source_integrity'] = imported.verify_files(directory, spec)
        expected = read_json(args.environment)
        configure(expected, output, args.physical_card)
        # Original wrappers sometimes write profiler files relative to __file__.
        # Execute an exact private copy, so they cannot dirty shared repository sources.
        execution = output / 'execution'
        execution.mkdir()
        for item in spec['files']:
            dest = execution / item['path']
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(directory / item['path'], dest)
        imported.verify_files(execution, spec)
        save(execution / 'test.json', spec)
        for key, name in (('TRITON_HOME', 'triton-home'), ('XDG_CACHE_HOME', 'xdg'),
                          ('TORCH_EXTENSIONS_DIR', 'torch-extensions')):
            folder = output / 'cache' / name
            folder.mkdir()
            os.environ[key] = str(folder)
        os.environ['BENCH_NPU_ID'] = '0'
        for key, value in spec.get('required_environment', {}).items():
            if not key.startswith('FLA_'):
                raise ValueError('Unexpected per-test environment override: ' + key)
            os.environ[key] = value
        os.chdir(output)
        result['environment'] = audit(expected)
        import torch
        torch.set_num_threads(4)
        phase = 'device_gate'
        with Monitor(args.physical_card, identity(), output / 'device.jsonl') as monitor:
            if torch.npu.device_count() != 1:
                raise RuntimeError('Expected one explicitly selected logical NPU')
            torch.npu.set_device(0)
            phase = 'execute'
            result['execution_attempted'] = True
            import numpy as np
            random.seed(spec['seed'])
            np.random.seed(spec['seed'])
            torch.manual_seed(spec['seed'])
            result['seed'] = spec['seed']
            worker_args = SimpleNamespace(output=output, samples=args.samples, warmup=args.warmup,
                correctness_only=args.correctness_only or spec['correctness_kind'] != 'assertions',
                block_factor=args.factor if args.variant == 'block' else None, axis=args.axis,
                target_launch_policy=target_launch_policy,
                test_module_policy=lambda module: select_input_config(module, config.get('input_selection')))
            original_measure = imported.measure
            def measure(capture, dest, samples, warmup):
                return measure_rounds(original_measure, capture, dest, samples, warmup, args.rounds)
            first = len(monitor.samples) - 1
            with patch.object(imported, 'measure', measure):
                imported.execute(execution, spec, worker_args, result)
            if spec['correctness_kind'] != 'assertions' and result['status'] in GOOD:
                result['status'] = 'accuracy_unvalidated'
            monitor.sample()
        # Include the final monitor sample, not just samples before context exit.
        if result['timing']:
            eligible = monitor.eligible_since(first)
            result['timing']['performance_eligible'] = eligible
            if not eligible:
                result['status'] = 'timing_rejected_interference'
    except Exception as error:
        status = ('infrastructure_error' if phase == 'setup' else 'device_unavailable' if phase == 'device_gate'
                  else 'compile_failed' if compile_error(error) else 'accuracy_failed'
                  if isinstance(error, AssertionError) else 'runtime_failed')
        result.update(status=status, error=traceback.format_exc())
        if result['timing']:
            result['timing']['performance_eligible'] = False
    finally:
        os.chdir(cwd)
        result.update(completed=True, ended_epoch=time.time())
        save(output / 'result.json', result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result['status'] in GOOD else 1


def lock_root():
    root = Path('/tmp') / f'triton-operator-benchmarks-{os.getuid()}'
    root.mkdir(mode=0o700, exist_ok=True)
    if root.is_symlink() or root.stat().st_uid != os.getuid():
        raise RuntimeError('Invalid per-user lock root')
    return root


@contextmanager
def batch_locks(args):
    if sys.platform != 'linux':
        raise RuntimeError('Device batches require Linux; verify and plan work on any host')
    import fcntl
    root = lock_root()
    paths = [root / f'card-{args.physical_card}.lock']
    paths += [p.resolve() for p in (args.task_lock, args.card_lock) if p is not None]
    if len(set(paths)) != len(paths):
        raise ValueError('Task and card lock paths must differ')
    with ExitStack() as stack:
        handles = []
        for path in sorted(paths):
            handle = stack.enter_context(path.open('a'))
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError('Another task owns lock: ' + str(path)) from error
            handles.append(handle)
        yield handles


def stop_worker(process):
    """Signal only the process group created by this controller."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)
    # A compiler/profiler grandchild might outlive its direct parent.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def worker_command(args, row, config, variant, attempt, fds):
    command = [sys.executable, '-I', str(ROOT / 'tools/run_fla_vv.py'), '_worker',
               '--cases', str(row['index']), '--test-id', config['test_id'], '--variant', variant,
               '--physical-card', str(args.physical_card), '--environment', str(args.environment.resolve()),
               '--output', str(attempt), '--factor', str(args.factor), '--axis', str(args.axis),
               '--rounds', str(args.rounds), '--samples', str(args.samples), '--warmup', str(args.warmup),
               '--lock-fds', ','.join(map(str, fds))]
    if args.correctness_only:
        command.append('--correctness-only')
    return command


def launch_trial(args, row, config, variant, dest, fds):
    attempts = sorted(dest.glob('attempt-*')) if dest.exists() else []
    # Never silently rerun an attempt orphaned before its durable terminal result.
    if attempts:
        previous = attempts[-1] / 'result.json'
        if not previous.exists() or not read_json(previous).get('completed'):
            raise RuntimeError('Incomplete attempt needs review before retry: ' + str(attempts[-1]))
        old = read_json(previous)
        if old['status'] not in RETRYABLE:
            return old, previous
    dest.mkdir(parents=True, exist_ok=True)
    attempt = dest / f'attempt-{len(attempts) + 1:03d}'
    command = worker_command(args, row, config, variant, attempt, fds)
    with (dest / f'{attempt.name}.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   pass_fds=tuple(fds), start_new_session=True)
        try:
            process.wait(timeout=args.worker_timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            stop_worker(process)
            attempt.mkdir(exist_ok=True)
            path = attempt / 'result.json'
            prior = read_json(path) if path.exists() else {}
            if path.exists():
                shutil.copyfile(path, attempt / 'result-before-stop.json')
            status = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'worker_timeout'
            prior.update(completed=True, status=status, timing=None,
                         ended_epoch=time.time(), supervisor_stopped_group=process.pid)
            save(path, prior)
            if isinstance(error, KeyboardInterrupt):
                raise
    path = attempt / 'result.json'
    if not path.exists() or not read_json(path).get('completed'):
        # Preserve partial worker output and refuse a false successful completion.
        raise RuntimeError('Worker exited without a terminal result: ' + str(attempt))
    return read_json(path), path


def eligible_us(record):
    if not record or record.get('status') != 'measured':
        return None
    timing = record.get('timing') or {}
    value = timing.get('kernel_us')
    if timing.get('performance_eligible') is not True or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        return None
    return value


def trial_key(row, config, variant):
    return f"{row['id']}/{config['test_id']}/{variant}"


def export(output, summary, rows):
    fields = ['vv_index', 'excel_row', 'operator_id', 'kernel', 'shape_dtype', 'type', 'test_id',
              'config', 'baseline_status', 'baseline_us', 'block_status', 'block_us', 'speedup',
              'baseline_result', 'block_result']
    path = output / 'performance.csv'
    temp = path.with_suffix('.csv.tmp')
    with temp.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            case = load_case(row)
            for config in case['configurations']:
                base = summary['trials'].get(trial_key(row, config, 'baseline'), {})
                block = summary['trials'].get(trial_key(row, config, 'block'), {})
                b, c = eligible_us(base), eligible_us(block)
                writer.writerow(dict(vv_index=row['index'], excel_row=row['row'], operator_id=row['operator_id'],
                    kernel=row['kernel'], shape_dtype=case['shape_dtype'], type='VV', test_id=config['test_id'],
                    config=json.dumps(config['config'], sort_keys=True), baseline_status=base.get('status', 'pending'),
                    baseline_us=b, block_status=block.get('status', 'pending' if summary['run_config']['factor'] > 1 else 'not_requested'),
                    block_us=c, speedup=b / c if b is not None and c is not None else None,
                    baseline_result=base.get('result_path'), block_result=block.get('result_path')))
    temp.replace(path)
    summary['counts'] = dict(Counter(r['status'] for r in summary['trials'].values()))
    summary['updated_epoch'] = time.time()
    save(output / 'summary.json', summary)


def run_config(args, rows):
    return dict(physical_card=args.physical_card, factor=args.factor, axis=args.axis,
        rounds=args.rounds, samples=args.samples, warmup=args.warmup, correctness_only=args.correctness_only,
        environment=str(args.environment.resolve()), environment_sha256=sha(args.environment),
        task_lock=str(args.task_lock.resolve()) if args.task_lock else None,
        card_lock=str(args.card_lock.resolve()) if args.card_lock else None,
        worker_timeout=args.worker_timeout, manifest_sha256=sha(GROUP / 'vv-manifest.json'),
        cases_sha256={path: sha(ROOT / path / 'cases-vv.json') for path in sorted({r['path'] for r in rows})},
        runner_sha256={p.relative_to(ROOT).as_posix(): sha(p)
                       for p in sorted([*ROOT.glob('opbench/*.py'), ROOT / 'tools/run_fla_vv.py'])})


def validate_resume(summary, rows, config):
    if summary['selection'] != [r['row'] for r in rows] or summary['run_config'] != config:
        raise ValueError('Resume requires identical selection, environment, source and run options')


def dispatch(args, output, rows, summary, fds):
    for row in rows:
        case = load_case(row)
        for config in case['configurations']:
            load_test(case, config)
            variants = ('baseline', 'block') if args.factor > 1 else ('baseline',)
            for variant in variants:
                key = trial_key(row, config, variant)
                prior = summary['trials'].get(key)
                if prior and prior['status'] not in RETRYABLE:
                    continue
                if variant == 'block':
                    base = summary['trials'][trial_key(row, config, 'baseline')]
                    if base['status'] not in GOOD:
                        summary['trials'][key] = dict(status='skipped_baseline_failed', timing=None,
                                                     baseline_status=base['status'])
                        export(output, summary, rows)
                        continue
                print(f"VV {row['index']}/317, Excel {row['row']}, {row['kernel']}, {variant}", flush=True)
                record, path = launch_trial(args, row, config, variant, output / key, fds)
                summary['trials'][key] = dict(record, result_path=path.relative_to(output).as_posix())
                export(output, summary, rows)
                if record['status'] in RETRYABLE:
                    summary['state'] = 'waiting_for_device' if record['status'] == 'device_unavailable' else 'review_required'
                    return 2
    summary['completed'] = True
    summary['all_passed'] = all(r['status'] in GOOD for r in summary['trials'].values())
    summary['state'] = 'finished' if summary['all_passed'] else 'finished_with_failures'
    return 0


def run_batch(args):
    rows = selected(args.cases, args.rows, args.operator)
    config = run_config(args, rows)
    output = args.output.resolve()
    with batch_locks(args) as handles:
        if output.exists() and not args.resume:
            raise FileExistsError('Output exists; use --resume with the same selection and options')
        if args.resume and not (output / 'summary.json').is_file():
            raise FileNotFoundError('Resume requires an existing summary.json')
        output.mkdir(parents=True, exist_ok=True)
        # Cross-card invocations must not overwrite the same output, either.
        import fcntl
        with (output / '.batch.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            handles = handles + [lock]
            if args.resume:
                summary = read_json(output / 'summary.json')
                validate_resume(summary, rows, config)
            else:
                summary = dict(schema_version=1, completed=False, selection=[r['row'] for r in rows],
                               run_config=config, trials={}, started_epoch=time.time())
            summary['state'] = 'running'
            export(output, summary, rows)
            try:
                return dispatch(args, output, rows, summary, [h.fileno() for h in handles])
            except KeyboardInterrupt:
                summary.update(completed=False, state='interrupted')
                return 130
            except Exception:
                summary.update(completed=False, state='review_required', error=traceback.format_exc())
                raise
            finally:
                export(output, summary, rows)
