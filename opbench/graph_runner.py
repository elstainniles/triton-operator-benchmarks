"""Generated benchmark cases: check outputs, then profile only direct Triton calls."""
from datetime import datetime, timezone
import gc
import os
from pathlib import Path
import traceback

from .catalog import module, plans, read_json, resolve
from .device import Monitor, identity
from .environment import audit, configure, digest
from .graph_tests import load_kernels, verify_graph_suite
from .profile import analyze
from .runner import compile_error, save


def measure(case, kernels, symbol, profile_dir, runtime_name):
    import torch
    import torch_npu
    labels = []
    exp = torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1, l2_cache=False)
    with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            record_shapes=False, profile_memory=False, with_stack=False, experimental_config=exp,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(profile_dir))) as prof:
        for round_id in range(3):
            for repeat in range(5):
                case.prepare()
                torch.npu.synchronize()
                label = f'opbench::baseline::{round_id}::{repeat}'
                with torch.profiler.record_function(label):
                    case.launch(kernels, symbol)
                torch.npu.synchronize()
                labels.append(label)
            prof.step()
    return analyze(profile_dir, labels, runtime_name, allow_unlabelled_kernels=True)


def run(args):
    import fcntl
    entry, path = resolve(args.operator)
    schedule = plans(args.operator, args.suite, args.variants)
    if getattr(args, 'case_id', None):
        wanted = set(args.case_id)
        available = {p['case']['id'] for p in schedule}
        if not wanted <= available:
            raise ValueError('Unknown or filtered case IDs: ' + ','.join(sorted(wanted-available)))
        schedule = [p for p in schedule if p['case']['id'] in wanted]
    if not schedule:
        raise ValueError('No matching generated cases')
    metadata = read_json(path / 'operator.json')
    source_audit = verify_graph_suite(path)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = dict(schema_version=1, operator=entry, completed=False, all_passed=False, status='pending',
                  source_audit=source_audit, source_revision=metadata['source']['revision'],
                  source_hashes={p.name:digest(p) for p in path.glob('*.py')},
                  input_origin='generated_benchmark', external_input_required=False,
                  compile_options={'enable_graph_optimize':False}, physical_card=args.physical_card,
                  started_utc=datetime.now(timezone.utc).isoformat(), cases=[])
    result['harness_sha256'] = {p.name:digest(p) for p in Path(__file__).parent.glob('*.py')}
    result['cases_sha256'] = digest(path/'cases.json')
    target = output / 'result.json'
    save(target, result)
    try:
        expected = read_json(args.environment)
        configure(expected, output, args.physical_card)
        lockroot = Path('/tmp') / f'triton-operator-benchmarks-{os.getuid()}'
        lockroot.mkdir(mode=0o700, exist_ok=True)
        if lockroot.is_symlink() or lockroot.stat().st_uid != os.getuid():
            raise RuntimeError('Invalid card lock directory')
        with (lockroot / f'card-{args.physical_card}.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            owner = identity()
            result['identity'] = owner
            with Monitor(args.physical_card, owner, output / 'device.jsonl') as monitor:
                result['environment'] = audit(expected)
                import torch
                import triton
                from triton.backends.ascend.compiler import AscendBackend
                torch.set_num_threads(4)
                if torch.npu.device_count() != 1:
                    raise RuntimeError('Expected exactly one selected NPU')
                torch.npu.set_device(0)
                backend = AscendBackend(triton.runtime.driver.active.get_current_target())
                if getattr(backend.parse_options({'enable_graph_optimize':False}), 'enable_graph_optimize', None) is not False:
                    raise RuntimeError('Backend does not support disabling graph optimization')
                result['backend_graph_option_verified'] = True
                result['device_name'] = torch.npu.get_device_name(0)
                kernels = load_kernels(path)
                for plan in schedule:
                    monitor.require_clear()
                    index = len(monitor.samples)-1
                    record = dict(case=plan['case'], status='pending', median_us=None)
                    result['cases'].append(record)
                    case = None
                    try:
                        case = module(entry, 'benchmark').make_case(plan['case'])
                        record.update(inputs=case.describe(), grid=list(case.grid),
                                      launch_options=dict(case.options, enable_graph_optimize=False))
                        case.prepare()
                        torch.npu.synchronize()
                        compiled = case.launch(kernels, metadata['kernel_symbol'])
                        torch.npu.synchronize()
                        compiled_options = getattr(getattr(compiled, 'metadata', None), 'enable_graph_optimize', None)
                        if compiled_options is not False:
                            raise RuntimeError('Compiled kernel metadata does not confirm graph optimization disabled')
                        record['compiled_graph_option'] = compiled_options
                        record['runtime_kernel_name'] = backend.pack_metadata(compiled.metadata)['kernel_name']
                        record['correctness'] = case.validate()
                        record['status'] = 'correct' if record['correctness']['passed'] else 'accuracy_failed'
                        if not record['correctness']['guards']:
                            raise RuntimeError('Output guard damaged')
                        if record['correctness']['passed'] and not args.correctness_only:
                            for _ in range(3):
                                case.prepare()
                                case.launch(kernels, metadata['kernel_symbol'])
                            torch.npu.synchronize()
                            monitor.require_clear()
                            if getattr(args, 'timing', 'profiler') == 'mspti':
                                from .mspti import measure_npu
                                profile = measure_npu([('baseline', lambda: case.launch(kernels, metadata['kernel_symbol']),
                                                        record['runtime_kernel_name'])], prepare=case.prepare)
                            else:
                                profile = measure(case, kernels, metadata['kernel_symbol'], output / ('profile-'+plan['case']['id']),
                                                  record['runtime_kernel_name'])
                            record['profile'] = profile
                            monitor.sample()
                            if not monitor.eligible_since(index):
                                record['status'] = 'timing_rejected_interference'
                                raise RuntimeError('Interfered measurement rejected')
                            record.update(status='measured', median_us=profile['variants']['baseline']['median_us'])
                    except Exception as exc:
                        record['error'] = traceback.format_exc()
                        if compile_error(exc):
                            record['status'] = 'compile_failed'
                        else:
                            if record['status'] != 'timing_rejected_interference':
                                record['status'] = 'runtime_failed'
                            raise
                    finally:
                        save(target, result)
                        print(f'{entry["id"]} {plan["case"]["id"]}: {record["status"]} {record["median_us"]}', flush=True)
                        del case
                        gc.collect()
                    monitor.require_clear()
                monitor.require_clear()
            passed = bool(result['cases']) and all(row['status'] in ('correct','measured') for row in result['cases'])
            result.update(completed=True, all_passed=passed, status='finished' if passed else 'finished_with_failures')
    except BaseException:
        result['error'] = traceback.format_exc()
        result['status'] = 'failed'
        raise
    finally:
        result['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save(target, result)
