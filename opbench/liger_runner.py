"""One exact second-batch kernel per process, correctness before target-only timing."""
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import traceback

from .catalog import plans, read_json, resolve
from .device import Monitor, identity
from .environment import audit, configure, digest
from .liger import GROUP, audit_dependency, load_kernel, verify_operator
from .runner import save


def failure_status(error, launched):
    text = str(error)
    if any(s in text for s in ('CompilationError', 'MLIRCompilationError', 'OutOfResources',
                               'MaxThreads out of range', "'dtype' object is not callable")):
        return 'compile_failed' if launched else 'blocked_by_dependency_compile'
    if 'AssertionError' in text:
        return 'accuracy_failed'
    return 'runtime_failed'


def run_case(path, meta, case, output, correctness_only, timing_method="profiler"):
    import pytest
    import torch
    import triton.runtime.jit

    name = meta['kernel_symbol']
    expected = (path/'kernels.py').resolve()
    captures, calls, reports = [], [], []

    class Capture:
        @pytest.hookimpl(hookwrapper=True)
        def pytest_runtest_call(self, item):
            original = triton.runtime.jit.JITFunction.run

            def record(jit, *args, grid, warmup, **kwargs):
                target = jit.fn.__name__ == name and Path(jit.fn.__code__.co_filename).resolve() == expected
                actual = {'grid': None if callable(grid) else grid}

                def grid_fn(values):
                    actual['grid'] = grid(values)
                    return actual['grid']

                if target and not warmup:
                    calls.append(name)
                result = original(jit, *args, grid=grid_fn if target and callable(grid) else grid,
                                  warmup=warmup, **kwargs)
                if target and not warmup:
                    captures.append(dict(jit=jit, args=args, kwargs=kwargs, grid=grid, actual_grid=actual['grid']))
                return result

            triton.runtime.jit.JITFunction.run = record
            try:
                yield
            finally:
                triton.runtime.jit.JITFunction.run = original

        def pytest_runtest_logreport(self, report):
            reports.append(dict(when=report.when, outcome=report.outcome, details=report.longreprtext))

    config = GROUP/'pytest.ini'
    nodeid = str(path/'test_case.py')+'::'+case['nodeid'].split('::',1)[1]
    rc = pytest.main(['-c', str(config), '--confcutdir', str(GROUP), '-p', 'no:cacheprovider',
                      '--import-mode=importlib', nodeid], plugins=[Capture()])
    result = dict(case=case, status='pending', median_us=None, pytest_exit=int(rc),
                  reports=reports, target_launch_attempts=len(calls), captured_launches=len(captures))
    # Require setup, call AND teardown success; never time skipped/missing tests.
    if rc != 0 or len(reports) != 3 or any(r['outcome'] != 'passed' for r in reports):
        if any(r['outcome']=='skipped' for r in reports):
            result['status'] = 'skipped'
        else:
            result['status'] = failure_status('\n'.join(r['details'] for r in reports), bool(calls))
        return result
    result['correctness'] = dict(passed=True, reference=meta['correctness']['reference'])
    if not captures:
        result['status'] = 'target_not_launched'
        return result
    launch = captures[-1]  # final upstream autotune winner, as in the original batch

    def describe(value):
        if isinstance(value, torch.Tensor):
            return dict(shape=list(value.shape),dtype=str(value.dtype),stride=list(value.stride()))
        return repr(value)

    result.update(grid=launch['actual_grid'], args=[describe(x) for x in launch['args']],
                  kwargs={k:describe(v) for k,v in launch['kwargs'].items()},
                  constexpr={k:v for k,v in launch['kwargs'].items() if isinstance(v,(str,int,float,bool))})
    if correctness_only:
        result['status'] = 'correct'
        return result

    def replay():
        launch['jit'].run(*launch['args'], grid=launch['grid'], warmup=False, **launch['kwargs'])

    from .liger_profile import measure
    try:
        timing = measure(replay, name, str(expected), output, method=timing_method)
        result['timing'] = timing
        if timing['status'] == 'MEASURED_KERNEL':
            result.update(status='measured', median_us=timing['us'])
        else:
            result['status'] = 'timing_rejected_interference'
    except Exception:
        result.update(status='timing_failed', error=traceback.format_exc())
    return result


def run(args):
    import fcntl
    entry, path = resolve(args.operator)
    meta, _ = verify_operator(path)
    schedule = plans(args.operator, args.suite, args.variants)
    if args.case_id:
        raise ValueError('Liger uses its frozen second-batch case; --case-id is unnecessary')
    if len(schedule) != 1:
        raise ValueError('No matching Liger second-batch configuration')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = dict(schema_version=1, origin='new_runner', operator=entry, completed=False,
                  all_passed=False, cases=[], started_utc=datetime.now(timezone.utc).isoformat(),
                  physical_card=args.physical_card, source_revision=meta['source']['revision'],
                  source_hashes={p.name:digest(p) for p in path.glob('*.py')},
                  harness_sha256={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},
                  cases_sha256=digest(path/'cases.json'))
    result_path = output/'result.json'
    save(result_path, result)
    try:
        expected = read_json(args.environment)
        configure(expected, output, args.physical_card)
        if any(k.startswith('LIGER_KERNEL_IMPL') for k in os.environ):
            raise RuntimeError('Remove inherited Liger implementation overrides')
        os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        os.environ['TRITON_KERNEL_DUMP'] = '1'
        # All compiler/cache output stays inside this unique run.
        for key, folder in [('TRITON_HOME','triton-home'),('XDG_CACHE_HOME','xdg'),
                            ('TORCH_EXTENSIONS_DIR','torch-extensions')]:
            location = output/'cache'/folder
            location.mkdir()
            os.environ[key] = str(location)
        lockroot = Path('/tmp')/f'triton-operator-benchmarks-{os.getuid()}'
        lockroot.mkdir(mode=0o700, exist_ok=True)
        if lockroot.is_symlink() or lockroot.stat().st_uid != os.getuid():
            raise RuntimeError('Invalid card lock directory')
        with (lockroot/f'card-{args.physical_card}.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            owner = identity()
            result['identity'] = owner
            with Monitor(args.physical_card, owner, output/'device.jsonl') as monitor:
                result['environment'] = audit(expected)
                # CANN platform helpers are required by upstream NPU launchers.
                cann_python = Path(expected['cann_root'])/'python/site-packages'
                if not (cann_python/'tbe/common/platform/__init__.py').is_file():
                    raise RuntimeError('Selected CANN is missing tbe platform helpers')
                sys.path.insert(0, str(cann_python))
                result['liger_dependency'] = audit_dependency()
                import torch
                torch.set_num_threads(8)
                if torch.npu.device_count() != 1:
                    raise RuntimeError('Expected one visible NPU')
                torch.npu.set_device(0)
                result['device_name'] = torch.npu.get_device_name(0)
                result['binding'] = load_kernel(path, meta)
                sys.path.insert(0, str(GROUP/'support'))
                monitor.require_clear()
                start = len(monitor.samples)-1
                record = run_case(path, meta, schedule[0]['case'], output, args.correctness_only,
                                  getattr(args, 'timing', 'profiler'))
                result['cases'].append(record)
                monitor.sample()
                if not monitor.eligible_since(start):
                    record.update(status='timing_rejected_interference',median_us=None)
                save(result_path, result)
                # Compile/numeric/skip failures are terminal per kernel and batchable.
                # Unknown runtime, attribution, or occupancy faults stop dispatch.
                if record['status'] in ('runtime_failed','timing_failed','timing_rejected_interference','target_not_launched'):
                    raise RuntimeError('Review '+record['status']+' before continuing; evidence saved')
                monitor.require_clear()
            result.update(completed=True, all_passed=record['status'] in ('measured','correct'))
    except BaseException:
        result['error'] = traceback.format_exc()
        raise
    finally:
        result['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save(result_path,result)
