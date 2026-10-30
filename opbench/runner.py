"""Ascend single-operator runner; verified scopes are recorded in docs/VALIDATION.md."""
from collections import defaultdict
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import random
import time
import traceback

from .catalog import module, plans, read_json, resolve
from .device import Monitor, identity
from .environment import audit, configure, digest
from .profile import analyze


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def compile_error(exc):
    return type(exc).__name__ in {'CompilationError', 'MLIRCompilationError', 'OutOfResources',
                                 'CompileTimeAssertionFailure', 'PTXASError'}


def run(args):
    import fcntl
    entry, op_path = resolve(args.operator)
    metadata = read_json(op_path / 'operator.json')
    schedule = plans(args.operator, args.suite, args.variants)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = dict(schema_version=1, origin='new_runner', operator=entry, completed=False, all_passed=False,
                  started_utc=datetime.now(timezone.utc).isoformat(), physical_card=args.physical_card,
                  source_hashes={str(p.relative_to(op_path)):digest(p) for p in op_path.glob('*.py')},
                  cases=[], run_error=None)
    result_path = output / 'result.json'
    save(result_path, result)
    try:
        expected = read_json(args.environment)
        configure(expected, output, args.physical_card)
        # Cooperating runs under the same user share this lock across checkouts.
        # npu-smi checks additionally protect against non-cooperating users/containers.
        lockroot = Path('/tmp') / f'triton-operator-benchmarks-{os.getuid()}'
        lockroot.mkdir(mode=0o700, exist_ok=True)
        if lockroot.stat().st_uid != os.getuid() or lockroot.is_symlink():
            raise RuntimeError('Invalid per-user lock directory')
        with (lockroot / f'card-{args.physical_card}.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            owner = identity()
            result['identity'] = owner
            with Monitor(args.physical_card, owner, output / 'device.jsonl') as monitor:
                result['environment'] = audit(expected)
                import torch
                import torch_npu
                torch.set_num_threads(4)
                if torch.npu.device_count() != 1:
                    raise RuntimeError('Expected exactly one explicitly visible NPU')
                torch.npu.set_device(0)
                result['device_name'] = torch.npu.get_device_name(0)
                save(result_path, result)
                groups = defaultdict(list)
                for plan in schedule:
                    groups[plan['case']['id']].append(plan)
                for case_id, group in groups.items():
                    monitor.require_clear()
                    sample_index = len(monitor.samples) - 1
                    config = group[0]['case']
                    case = module(entry, 'case').make_case(config['shape'], getattr(torch, config['dtype']), config['seed'])
                    record = dict(case=config, variants=[], timing_eligible=False)
                    result['cases'].append(record)
                    accepted = []
                    for plan in group:
                        monitor.require_clear()
                        item = dict(variant_id=plan['variant_id'], config=plan['config'], grid=plan['grid'],
                                    launch_options=plan['launch_options'], status='pending', median_us=None)
                        record['variants'].append(item)
                        fn = lambda p=plan: module(entry, 'launch').launch(case, p['config'])
                        before = time.perf_counter()
                        try:
                            # A previous candidate's values must not hide missing stores.
                            for actual in case.actual:
                                actual.fill_(float('nan'))
                            torch.npu.synchronize()
                            compiled = fn()
                            torch.npu.synchronize()
                            if getattr(args, 'timing', 'profiler') == 'mspti':
                                import triton
                                from triton.backends.ascend.compiler import AscendBackend
                                backend = AscendBackend(triton.runtime.driver.active.get_current_target())
                                item['runtime_kernel_name'] = backend.pack_metadata(compiled.metadata)['kernel_name']
                            item['correctness'] = case.validate()
                            item['status'] = 'correct' if item['correctness']['passed'] else 'accuracy_failed'
                            if not item['correctness']['guards']:
                                raise RuntimeError('Output guard corruption; aborting device run')
                            if item['status'] == 'correct':
                                for _ in range(3):
                                    fn()
                                torch.npu.synchronize()
                                accepted.append((item, fn))
                        except Exception as exc:
                            item['error'] = traceback.format_exc()
                            item['status'] = 'compile_failed' if compile_error(exc) else 'runtime_failed'
                            if item['status'] == 'runtime_failed':
                                raise
                        finally:
                            item['cold_call_seconds'] = time.perf_counter() - before
                            save(result_path, result)
                        print(json.dumps(dict(case=case_id, variant=item['variant_id'], status=item['status'])), flush=True)
                    monitor.require_clear()
                    if not args.correctness_only and accepted:
                        profile_dir = output / ('profile-' + case_id)
                        labels = []
                        if getattr(args, 'timing', 'profiler') == 'mspti':
                            from .mspti import measure_npu
                            profile = measure_npu([(item['variant_id'], fn, item['runtime_kernel_name'])
                                                   for item, fn in accepted])
                        else:
                            exp = torch_npu.profiler._ExperimentalConfig(
                                profiler_level=torch_npu.profiler.ProfilerLevel.Level1, l2_cache=False)
                            rng = random.Random(20261026)
                            with torch_npu.profiler.profile(
                                activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                                record_shapes=False, profile_memory=False, with_stack=False,
                                experimental_config=exp,
                                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(profile_dir))) as prof:
                                for round_id in range(3):
                                    order = list(accepted)
                                    rng.shuffle(order)
                                    for item, fn in order:
                                        for repeat in range(5):
                                            label = f'opbench::{item["variant_id"]}::{round_id}::{repeat}'
                                            torch.npu.synchronize()
                                            with torch.profiler.record_function(label):
                                                fn()
                                            torch.npu.synchronize()
                                            labels.append(label)
                                        prof.step()
                            record['profile_sequence'] = labels
                            profile = analyze(profile_dir, labels, metadata['kernel_symbol'])
                        monitor.sample()
                        record['profile'] = profile
                        record['timing_eligible'] = monitor.eligible_since(sample_index)
                        for item, _ in accepted:
                            measurement = profile['variants'][item['variant_id']]
                            item['raw_timing'] = measurement
                            if record['timing_eligible']:
                                item.update(status='measured', median_us=measurement['median_us'])
                            else:
                                item['status'] = 'timing_rejected_interference'
                        save(result_path, result)
                        if not record['timing_eligible']:
                            raise RuntimeError('Device monitoring rejected timing; raw evidence retained')
                    elif not monitor.eligible_since(sample_index):
                        raise RuntimeError('Device conflict during correctness checks')
                    save(result_path, result)
                    del case
                    accepted.clear()
                    gc.collect()
                monitor.require_clear()
            result['completed'] = True
            result['all_passed'] = bool(result['cases']) and all(v['status'] in ('correct','measured')
                                       for c in result['cases'] for v in c['variants'])
    except BaseException:
        result['run_error'] = traceback.format_exc()
        raise
    finally:
        result['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save(result_path, result)
