"""Run an imported FLA test with generated in-memory inputs and CPU references."""
import argparse
from contextlib import ExitStack, contextmanager
import csv
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys
import traceback
from unittest.mock import patch

from .fla_fixtures import use_fixtures

ROOT = Path(__file__).resolve().parents[1]
GROUP = ROOT / 'operators/flash_linear_attention'


def catalog():
    return json.loads((GROUP / 'test-sources.json').read_text())['tests']


def select(operator=None, case=None):
    rows = [r for r in catalog() if (operator is None or r['operator_id'] == operator)
            and (case is None or case in (r['case_id'], r['id']))]
    return rows


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def local_imports(directory, search):
    old_path, old_modules = list(sys.path), dict(sys.modules)
    try:
        sys.path[:0] = [str(path) for path in search]
        yield
    finally:
        sys.path[:] = old_path
        for name, module in list(sys.modules.items()):
            file = getattr(module, '__file__', None)
            if file and Path(file).resolve().is_relative_to(directory.resolve()):
                if name in old_modules:
                    sys.modules[name] = old_modules[name]
                else:
                    del sys.modules[name]


@contextmanager
def fixture_data(directory, spec):
    provider = directory / spec['provider']
    with local_imports(provider.parent, [provider.parent]):
        module = load(provider, '_fla_generated_fixture')
        fixtures = module.make_fixtures()
        # Map the old test's logical key to the generated payload; no disk fallback.
        from .fla_fixtures import lookup
        for original, target in spec.get('data_path_map', {}).items():
            try:
                lookup(fixtures, original)
                continue
            except KeyError:
                pass
            try:
                fixtures[original] = lookup(fixtures, target)
            except KeyError:
                # Some original maps alias inputs and outputs to one file. The
                # provider's explicit independently calculated output takes priority.
                pass
        with use_fixtures(fixtures):
            yield fixtures


def describe(value):
    import torch
    if isinstance(value, torch.Tensor):
        return dict(shape=list(value.shape), dtype=str(value.dtype), stride=list(value.stride()))
    if isinstance(value, dict):
        return {str(k): describe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [describe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def verify_files(directory, spec):
    from .fla_test_sources import inside, digest
    for record in spec['files']:
        path = inside(directory, record['path'])
        if digest(path) != record['sha256']:
            raise ValueError('Imported FLA test source changed: ' + record['path'])
    return dict(files_verified=len(spec['files']), test_manifest='test.json')


def block_status(report, factor, axis, grid, compile_report=None):
    """Accept only the branch's actual runtime decision, including its schedule."""
    if isinstance(compile_report, dict) and compile_report.get('compile_status') == 'REJECTED':
        return 'compile_rejected'
    if not isinstance(report, dict):
        return 'block_not_confirmed_applied'
    status = report.get('runtime_status')
    if status == 'BASELINE_SELECTED':
        reason = str(report.get('reason', ''))
        return 'no_merge_opportunity' if 'SELECTED_GRID_AXIS_HAS_LESS' in reason else 'runtime_fallback'
    if status != 'APPLIED':
        return 'block_not_confirmed_applied'
    original = list(grid) + [1] * (3 - len(grid))
    batch = list(original)
    batch[axis] = (batch[axis] + factor - 1) // factor
    if (report.get('owner') != 'block_batching' or report.get('requested_factor') != factor
            or report.get('applied_factor') != factor or report.get('selected_axis') != axis
            or list(report.get('original_grid', ())) != original
            or list(report.get('batch_grid', ())) != batch or not report.get('plan_id')):
        return 'block_runtime_evidence_mismatch'
    return 'applied'


def runtime_kernel_name(compiled, target):
    report = getattr(compiled, 'last_launch', None)
    active = (getattr(compiled, 'candidate', None)
              if isinstance(report, dict) and report.get('runtime_status') == 'APPLIED' else compiled)
    packed = getattr(active, 'packed_metadata', None)
    if isinstance(packed, dict) and packed.get('kernel_name'):
        return packed['kernel_name']
    from triton.backends.ascend.compiler import AscendBackend
    import triton
    backend = AscendBackend(triton.runtime.driver.active.get_current_target())
    return backend.pack_metadata(active.metadata)['kernel_name']


def measure(capture, output, samples, warmup):
    import torch
    import torch_npu
    snapshot = capture['snapshot']
    def replay():
        compiled = capture['jit'].run(*capture['args'], grid=capture['grid'], warmup=False,
                                      **capture['kwargs'])
        if capture.get('block_factor'):
            report = getattr(compiled, 'last_launch', None)
            status = block_status(report, capture['block_factor'], capture['axis'], capture['actual_grid'],
                                  getattr(compiled, 'block_batching', None))
            if status != 'applied':
                raise RuntimeError('Block replay was not applied: ' + status + ': ' + repr(report))
        return compiled
    durations, evidence = [], []
    try:
        for _ in range(warmup):
            snapshot.restore_pre(); replay(); torch.npu.synchronize()
            snapshot.verify_post()
        for index in range(samples):
            snapshot.restore_pre()
            folder = output / 'profiles' / str(index)
            with torch_npu.profiler.profile(
                    activities=[torch_npu.profiler.ProfilerActivity.CPU,
                                torch_npu.profiler.ProfilerActivity.NPU],
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(folder)),
                    record_shapes=False, profile_memory=False, with_stack=False) as profile:
                replay()
                torch.npu.synchronize()
                profile.step()
            snapshot.verify_post()
            files = list(folder.rglob('kernel_details.csv'))
            if len(files) != 1:
                raise RuntimeError(f'Expected one kernel timing CSV, found {len(files)}')
            with files[0].open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            rows = [row for row in rows if row.get('Name') == capture['name']]
            if len(rows) != 1:
                raise RuntimeError(f'Expected one target device sample, found {len(rows)}')
            value = float(rows[0]['Duration(us)'])
            if not math.isfinite(value) or not value > 0:
                raise RuntimeError('Invalid device duration')
            durations.append(value)
            evidence.append(dict(csv=str(files[0].relative_to(output)),
                sha256=hashlib.sha256(files[0].read_bytes()).hexdigest(), row=rows[0]))
    finally:
        snapshot.restore_post()
    return dict(metric='target_kernel_device_us', mean_us=statistics.mean(durations),
                samples_us=durations, sample_count=samples, warmup=warmup, evidence=evidence,
                state_restore_checks=snapshot.checks)


def execute(directory, spec, args, result):
    import torch
    import torch_npu
    import triton
    from triton.runtime.autotuner import Autotuner
    from .fla_assertions import OriginalAssertions
    from .fla_replay import StorageState

    target = spec['kernel']
    state = dict(recording=False, observed=False)
    captures = []
    launch_identities = set()
    paths = [directory / r['path'] for r in spec['files']
             if r['role'] in ('test', 'dependency') and r['path'].endswith('.py')]
    class Acceptance:
        close_passes = 0
        def __init__(self):
            self.failures, self.passed_source_assertions = [], []
        def failed(self, error, file, line):
            self.failures.append(dict(error=str(error), file=str(file), line=line))
    acceptance = Acceptance()
    observer = OriginalAssertions(torch, acceptance, paths, state)
    original_run = triton.runtime.jit.JITFunction.run
    original_tune = Autotuner.run
    original_bench = Autotuner._bench
    selection = spec.get('autotune_config') or {}

    def tune(autotuner, *values, **kwargs):
        jit = autotuner.fn
        while hasattr(jit, 'fn') and not hasattr(jit, 'arg_names'):
            jit = jit.fn
        if getattr(getattr(jit, 'fn', None), '__name__', None) == target and selection:
            options = {k: selection[k] for k in ('num_warps', 'num_stages', 'num_ctas', 'maxnreg') if k in selection}
            tile = selection.get('kwargs', {k: v for k, v in selection.items() if k not in options})
            existing = [config for config in autotuner.configs
                        if getattr(config, 'kwargs', None) == tile
                        and all(getattr(config, key, None) == value for key, value in options.items())]
            # Reuse the original object when possible, including its pre_hook.
            autotuner.configs = [existing[0] if existing else triton.Config(tile, **options)]
        return original_tune(autotuner, *values, **kwargs)

    def bench(autotuner, *values, **kwargs):
        old = state.get('tuning', False)
        state['tuning'] = True
        try:
            return original_bench(autotuner, *values, **kwargs)
        finally:
            state['tuning'] = old

    def record(jit, *values, grid, warmup, **kwargs):
        is_target = (jit.fn.__name__ == target and not state.get('tuning')
                     and Path(jit.fn.__code__.co_filename).resolve().is_relative_to(directory.resolve()))
        snapshot = None
        actual = {'grid': grid if not callable(grid) else None}
        def chosen_grid(meta):
            actual['grid'] = grid(meta)
            return actual['grid']
        if is_target and not warmup:
            state['observed'] = True
            if args.block_factor:
                kwargs = dict(kwargs, block_batching_factor=args.block_factor,
                              block_batching_axis=args.axis)
            tensors = [v for v in (*values, *kwargs.values()) if isinstance(v, torch.Tensor)]
            if not args.correctness_only:
                snapshot = StorageState(torch, tensors, torch.npu.synchronize, torch_npu.get_npu_format)
        compiled = original_run(jit, *values, grid=chosen_grid if callable(grid) else grid,
                                warmup=warmup, **kwargs)
        if is_target and not warmup:
            if snapshot: snapshot.capture_post()
            name = runtime_kernel_name(compiled, target)
            launch = getattr(compiled, 'last_launch', None)
            if hasattr(launch, '__dict__'): launch = vars(launch)
            identity = json.dumps(dict(kernel=name, grid=list(actual['grid']),
                                       arguments=describe(values), options=describe(kwargs)), sort_keys=True)
            launch_identities.add(hashlib.sha256(identity.encode()).hexdigest())
            captures[:] = [dict(jit=jit,args=values,kwargs=kwargs,grid=grid,
                actual_grid=actual['grid'],snapshot=snapshot,name=name,last_launch=launch,
                compile_report=getattr(compiled, 'block_batching', None),
                block_factor=args.block_factor,axis=args.axis)]
        return compiled

    with local_imports(directory, [directory / p for p in spec['pythonpath']]), fixture_data(directory, spec) as fixtures, ExitStack() as stack:
        result['generated_data'] = describe(fixtures)
        stack.enter_context(patch.object(triton.runtime.jit.JITFunction, 'run', record))
        stack.enter_context(patch.object(Autotuner, 'run', tune))
        stack.enter_context(patch.object(Autotuner, '_bench', bench))
        original_close = torch.testing.assert_close
        stack.enter_context(patch.object(torch.testing, 'assert_close',
            lambda *a, **kw: observer.close(original_close, *a, **kw)))
        for name in ('allclose', 'equal'):
            stack.enter_context(patch.object(torch, name, observer.compare(getattr(torch, name))))
        module = importlib.import_module(spec['module'])
        function = getattr(module, spec['function'])
        old_trace = sys.gettrace()
        state['recording'] = True
        sys.settrace(observer.trace)
        try:
            invocation = spec['invocation']
            returned = function(*invocation.get('args', []), **invocation.get('kwargs', {}))
            torch.npu.synchronize()
            if returned is False: raise AssertionError('Original test returned False')
        finally:
            sys.settrace(old_trace); state['recording'] = False
            result['correctness'] = observer.summary(sorted(launch_identities))
            if captures:
                last = captures[-1]
                result.update(runtime_kernel_name=last['name'],grid=list(last['actual_grid']),
                              arguments=describe(last['args']),launch_options=describe(last['kwargs']))
                if args.block_factor:
                    result['last_launch'] = describe(last['last_launch'])
                    result['block_compile_report'] = describe(last['compile_report'])
        if len(launch_identities) > 1:
            result['status'] = 'multiple_target_configurations'; return
        if result['correctness']['status'] != 'passed':
            result['status'] = 'accuracy_unvalidated'; return
        capture = captures[-1]
        result['status'] = 'correct'
        if args.block_factor:
            launch = capture['last_launch']
            result['last_launch'] = describe(launch)
            result['block_compile_report'] = describe(capture['compile_report'])
            status = block_status(launch,args.block_factor,args.axis,capture['actual_grid'],capture['compile_report'])
            if status != 'applied':
                result['status'] = status; return
        # Leave instrumentation before timing, so replay cannot overwrite its snapshot.
    if not args.correctness_only:
        result['timing'] = measure(capture, args.output, args.samples, args.warmup)
        result['status'] = 'measured'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operator')
    parser.add_argument('--case')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--check-fixtures', action='store_true')
    parser.add_argument('--physical-card', type=int)
    parser.add_argument('--environment', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--correctness-only', action='store_true')
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--block-factor', type=int)
    parser.add_argument('--axis', type=int, choices=(0,1,2))
    args = parser.parse_args(argv)
    rows = select(args.operator, args.case)
    if args.list:
        print(json.dumps(rows, indent=2, ensure_ascii=False)); return 0
    if len(rows) != 1:
        parser.error(f'Select one exact test variant with --case; matched {len(rows)}. Use --list.')
    if args.samples < 1 or args.warmup < 0: parser.error('Invalid sampling count')
    if args.block_factor is not None and (args.block_factor < 2 or args.axis is None):
        parser.error('Block tests require --block-factor >= 2 and an explicit --axis')
    info = ROOT / rows[0]['path']; spec = json.loads(info.read_text()); directory = info.parent
    if args.check_fixtures:
        verify_files(directory, spec)
        import torch
        torch.set_num_threads(4)
        # A regression to external serialization fails immediately on this CPU path.
        def no_tensor_file(*a, **kw): raise RuntimeError('External tensor serialization is forbidden')
        with patch.object(torch, 'load', no_tensor_file), patch.object(torch, 'save', no_tensor_file), fixture_data(directory, spec) as fixtures:
            print(json.dumps(describe(fixtures), indent=2))
        return 0
    if args.physical_card is None or args.environment is None or args.output is None:
        parser.error('Device runs require --physical-card, --environment and a fresh --output')
    args.output = args.output.resolve(); args.output.mkdir(parents=True, exist_ok=False)
    from .environment import configure, audit
    from .runner import compile_error, save
    from .device import Monitor, identity
    result = dict(case_id=spec['case_id'],test_id=spec['id'],kernel=spec['kernel'],status='pending',
                  external_input_required=False,timing=None,physical_card=args.physical_card,
                  source_manifest_sha256=hashlib.sha256(info.read_bytes()).hexdigest(),
                  requested_block_factor=args.block_factor,requested_axis=args.axis)
    original_dir = Path.cwd()
    try:
        expected = json.loads(args.environment.read_text())
        configure(expected,args.output,args.physical_card)
        result['source_integrity'] = verify_files(directory, spec)
        os.environ['BENCH_NPU_ID'] = '0'
        for key,value in spec.get('required_environment',{}).items(): os.environ[key] = value
        os.chdir(args.output)
        result['environment'] = audit(expected)
        import torch
        torch.set_num_threads(4)
        if torch.npu.device_count() != 1:
            raise RuntimeError('Expected one explicitly selected logical NPU')
        torch.npu.set_device(0)
        with Monitor(args.physical_card, identity(), args.output/'device.jsonl') as monitor:
            monitor.require_clear()
            first = len(monitor.samples)-1
            execute(directory, spec, args, result)
            monitor.sample()
            if result['timing'] and not monitor.eligible_since(first):
                result['status'] = 'timing_rejected_interference'
                result['timing']['performance_eligible'] = False
    except BaseException as error:
        result.update(status='compile_failed' if compile_error(error) else 'accuracy_failed'
                      if isinstance(error,AssertionError) else 'runtime_failed',error=traceback.format_exc())
    finally:
        os.chdir(original_dir)
        save(args.output/'result.json',result)
    print(json.dumps(result,indent=2,default=str))
    return 0 if result['status'] in ('correct','measured') else 1
