"""FLA generated baselines and explicit, device-free source archive records."""
from datetime import datetime, timezone
import gc
import os
from pathlib import Path
import traceback

from .catalog import module, plans, read_json, resolve
from .device import Monitor, identity
from .environment import audit, configure, digest
from .fla import verify_operator
from .graph_runner import measure
from .runner import compile_error, save


def run(args):
    entry, path = resolve(args.operator)
    metadata, _ = verify_operator(path)
    schedule = plans(args.operator, args.suite, args.variants)
    wanted = set(getattr(args, 'case_id', None) or [])
    if wanted:
        available = {plan['case']['id'] for plan in schedule if plan['case'] is not None}
        if not wanted <= available:
            raise ValueError('Unknown or filtered FLA case IDs: ' + ','.join(sorted(wanted - available)))
        schedule = [plan for plan in schedule if plan['case']['id'] in wanted]
    if not schedule:
        raise ValueError('No matching generated FLA cases')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = dict(schema_version=1, origin='new_runner', operator=entry, completed=False,
                  all_passed=False, status='pending', entry_mode=metadata['entry_mode'],
                  source_revision=metadata['source']['revision'],
                  source_hashes={p.name: digest(p) for p in path.glob('*.py')},
                  harness_sha256={p.name: digest(p) for p in Path(__file__).parent.glob('*.py')},
                  cases_sha256=digest(path / 'cases.json'),
                  started_utc=datetime.now(timezone.utc).isoformat(), cases=[])
    target = output / 'result.json'
    if metadata['entry_mode'] == 'source_archive':
        # No environment import, runtime setup, lock, or NPU query is needed.
        result.update(completed=True, status='not_executable', device_execution=False,
                      reason=metadata['adaptation_reason'], cases=[dict(
                          case=None, variant_id='baseline', status='not_executable',
                          median_us=None, reason=metadata['adaptation_reason'])],
                      finished_utc=datetime.now(timezone.utc).isoformat())
        save(target, result)
        print(entry['id'], 'not_executable:', result['reason'], flush=True)
        return result
    result.update(input_origin='generated_benchmark', external_input_required=False,
                  compile_options={'enable_graph_optimize': False}, physical_card=args.physical_card)
    save(target, result)
    try:
        import fcntl
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
                option = backend.parse_options({'enable_graph_optimize': False})
                if getattr(option, 'enable_graph_optimize', None) is not False:
                    raise RuntimeError('Backend does not support disabling graph optimization')
                result['device_name'] = torch.npu.get_device_name(0)
                kernels = module(entry, 'kernels')
                for plan in schedule:
                    monitor.require_clear()
                    index = len(monitor.samples) - 1
                    record = dict(case=plan['case'], variant_id='baseline', status='pending', median_us=None)
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
                        if getattr(getattr(compiled, 'metadata', None), 'enable_graph_optimize', None) is not False:
                            raise RuntimeError('Compiled metadata does not confirm graph optimization disabled')
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
                            profile = measure(case, kernels, metadata['kernel_symbol'],
                                              output / ('profile-' + plan['case']['id']),
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
                            if record['status'] not in ('timing_rejected_interference', 'accuracy_failed'):
                                record['status'] = 'runtime_failed'
                            raise
                    finally:
                        save(target, result)
                        print(entry['id'], plan['case']['id'], record['status'], record['median_us'], flush=True)
                        del case
                        gc.collect()
                    monitor.require_clear()
                monitor.require_clear()
            passed = bool(result['cases']) and all(row['status'] in ('correct', 'measured') for row in result['cases'])
            result.update(completed=True, all_passed=passed, status='finished' if passed else 'finished_with_failures')
    except BaseException:
        result.update(error=traceback.format_exc(), status='failed')
        raise
    finally:
        result['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save(target, result)
    return result
