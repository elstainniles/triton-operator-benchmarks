"""Guarded block-batching search over one original, fully restorable launch.

Importing this module does not import Torch/Triton or initialize a device.
"""
import hashlib
from collections import Counter
import json
import math
import os
import operator
from pathlib import Path
import statistics
import subprocess
import time
import traceback

from replay_state import ReplayMismatch, ReplayUnsupported, RestoreFailure, support_check, state_restored_replays, check_launch_hooks
from single_trace_timing import MissingTargetRows, measure_once as measure_single_trace_once, preliminary_samples_valid


class RuntimeFallback(RuntimeError):
    def __init__(self, report):
        super().__init__(str(report.get('reason', report.get('runtime_status', 'missing runtime report'))))
        self.report = report


class RuntimeContractError(RuntimeError):
    pass


class MeasurementError(RuntimeError):
    pass


class IncompleteDeviceTrace(MeasurementError):
    """A full replay round completed but one or more device traces are absent."""
    pass


def compare_ttir_structure(original, bundle, directory):
    """Compare parsed MLIR generic structure; retain originals and printer outputs."""
    if not isinstance(original,str) or not isinstance(bundle,str):
        raise RuntimeContractError('original or bundle TTIR is missing')
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    for name,data in (('original.ttir',original),('bundle.ttir',bundle)):
        (directory/name).write_text(data)
    import triton
    opt=Path(triton.__file__).resolve().parent/'_C'/'triton-opt'
    if not opt.is_file():raise RuntimeContractError('wheel MLIR parser is unavailable')
    canonical=[]
    for name in ('original','bundle'):
        r=subprocess.run([str(opt),'--allow-unregistered-dialect','--mlir-print-op-generic',str(directory/(name+'.ttir'))],capture_output=True,text=True,timeout=60)
        (directory/(name+'.generic.mlir')).write_text(r.stdout)
        (directory/(name+'.parser.stderr')).write_text(r.stderr)
        if r.returncode:raise RuntimeContractError('MLIR parser failed for '+name+': '+r.stderr)
        canonical.append(r.stdout)
    result={'equivalent':canonical[0]==canonical[1], 'method':'wheel triton-opt parsed generic MLIR; no transformations; debug locations excluded',
            'original_raw_sha256':_sha(original),'bundle_raw_sha256':_sha(bundle),
            'original_structural_sha256':_sha(canonical[0]),'bundle_structural_sha256':_sha(canonical[1]),
            'evidence_directory':str(directory)}
    (directory/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def _safe(value):
    if value is None or type(value) in (str, int, bool):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    return repr(value)


def _sha(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def compile_options(kernel, option_type):
    """Keep the backend's own metadata round-trip markers and public options."""
    metadata = kernel.metadata._asdict()
    fields = option_type.__dataclass_fields__
    if metadata.get('block_batching_factor', 1) != 1:
        raise ValueError('The original independent baseline must have block_batching_factor=1')
    # parse_options recognizes backend-derived metadata as an internal round trip.
    # Dropping these fields would reinterpret normalized options as user inputs.
    options = {name: value for name, value in metadata.items() if name in fields}
    options.update(block_batching_factor=1, block_batching_axis=-1, block_batching_layout='auto')
    return options


def verify_applied(report, axis, factor, original_grid):
    if not isinstance(report, dict):
        raise RuntimeContractError('Missing structured block batching runtime report')
    if (report.get('runtime_status') == 'BASELINE_SELECTED' and report.get('owner') == 'baseline'
            and report.get('applied_factor') == 1 and report.get('reason')):
        raise RuntimeFallback(report)
    if report.get('runtime_status') != 'APPLIED':
        raise RuntimeContractError('Unexpected runtime status: ' + str(report.get('runtime_status')))
    expected_grid = list(original_grid)
    expected_grid[axis] = (expected_grid[axis] + factor - 1) // factor
    if (report.get('owner') != 'block_batching' or report.get('requested_factor') != factor
            or report.get('applied_factor') != factor or report.get('selected_axis') != axis
            or list(report.get('original_grid', ())) != list(original_grid)
            or list(report.get('batch_grid', ())) != expected_grid or not report.get('plan_id')):
        raise RuntimeContractError('APPLIED report does not match requested factor, axis, grid or plan')


def original_compilation_target(kernel):
    """Use the original compiled kernel's target, not a fresh device guess."""
    from triton.backends.compiler import GPUTarget
    target = getattr(getattr(kernel, 'metadata', None), 'target', None)
    if not isinstance(target, GPUTarget):
        raise RuntimeContractError('original compiled kernel target metadata unavailable')
    legacy = getattr(kernel, 'target', target)
    if legacy != target:
        raise RuntimeContractError('kernel and metadata compilation targets disagree')
    return target


def measure_variant(storage, launch, directory, device_name, *, torch_npu, original_profile,
                    profile_rows, samples, warmup):
    """Accept the first complete round; retry only missing traces once, in full.

    Every attempt retains all trace files and parsed samples. An incomplete round
    contributes no timings to the accepted round, and timings never select retries.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    history = []
    maximum_attempts = 2

    def publish(result=None):
        summary = dict(result or {})
        summary.update(measurement_round_attempts=history,
                       maximum_measurement_attempts=maximum_attempts,
                       retry_policy='first_complete_round_missing_trace_only',
                       complete_rounds_combined=False,
                       retry_evidence_path=str(directory / 'measurement.json'))
        path = directory / 'measurement.json'
        temp = path.with_suffix('.json.tmp')
        temp.write_text(json.dumps(_safe(summary), indent=2) + '\n')
        os.replace(temp, path)
        return summary

    for index in range(1, maximum_attempts + 1):
        target = directory / ('attempt-%d' % index)
        record = dict(attempt=index, path=str(target), status='running')
        history.append(record)
        publish()
        try:
            result = _measure_variant_once(storage, launch, target, device_name,
                torch_npu=torch_npu, original_profile=original_profile,
                profile_rows=profile_rows, samples=samples, warmup=warmup)
        except Exception as error:
            record.update(status=('incomplete_device_trace' if isinstance(error, IncompleteDeviceTrace)
                                  else 'failed'), error_type=type(error).__name__,
                          error=str(error), traceback=traceback.format_exc(),
                          evidence_path=str(target / 'measurement.json'))
            publish()
            if isinstance(error, IncompleteDeviceTrace) and index < maximum_attempts:
                continue
            raise
        record.update(status='complete', evidence_path=str(target / 'measurement.json'),
                      raw_samples_us=list(result['durations_us']))
        result = publish(result)
        result['accepted_measurement_attempt'] = index
        return publish(result)


def _measure_variant_once(storage, launch, directory, device_name, *, torch_npu, original_profile,
                          profile_rows, samples, warmup):
    """Use the explicit single-trace protocol in this new, separately frozen bundle."""
    try:
        return measure_single_trace_once(storage, launch, directory, device_name,
            torch_npu=torch_npu, original_profile=original_profile,
            profile_rows=profile_rows, samples=samples, warmup=warmup)
    except MissingTargetRows as error:
        raise IncompleteDeviceTrace(str(error)) from error


def _measure_variant_once_legacy(storage, launch, directory, device_name, *, torch_npu, original_profile,
                                 profile_rows, samples, warmup):
    """One target kernel in each NPU profiler window; copies run outside RECORD."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    traces = []
    details = {}
    checks_before = storage.checks

    def profiler_factory():
        handler = torch_npu.profiler.tensorboard_trace_handler(str(directory), analyse_flag=False)

        def trace_ready(profiler):
            path = Path(profiler.prof_if.prof_path).resolve()
            if directory.resolve() not in path.parents:
                raise MeasurementError('Profiler trace escaped the measurement directory')
            traces.append(str(path))
            handler(profiler)

        config = torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization, l2_cache=False)
        return original_profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU], record_shapes=True,
            profile_memory=False, with_stack=False, experimental_config=config,
            schedule=torch_npu.profiler.schedule(wait=1, warmup=1, active=1, repeat=samples),
            on_trace_ready=trace_ready)

    try:
        state_restored_replays(storage, launch, profiler_factory, details, warmup=warmup, samples=samples)
        if len(traces) != samples or len(set(traces)) != samples:
            raise MeasurementError('Expected %d distinct profiling intervals' % samples)
        # torch_npu 2.7.1.post8 defines analyse in profiler.profiler but
        # does not re-export it from torch_npu.profiler.
        analyse = getattr(torch_npu.profiler, 'analyse', None)
        if analyse is None:
            from torch_npu.profiler.profiler import analyse
        analyse(str(directory), max_process_number=2)
        intervals = []
        interval_evidence = []
        missing = []
        fatal = []
        for path in traces:
            try:
                row = profile_rows(path, device_name)
            except Exception as error:
                entry = dict(trace_path=path, status='profile_parse_failed',
                             error_type=type(error).__name__, error=str(error))
                if isinstance(error, ValueError) and str(error) == 'Profiler produced no kernel_details.csv':
                    entry['status'] = 'missing_device_kernel_trace'
                    missing.append(path)
                else:
                    fatal.append(error)
                interval_evidence.append(entry)
            else:
                intervals.append(row)
                interval_evidence.append(dict(trace_path=path, status='parsed', result=row))
        details.update(intervals=intervals, interval_evidence=interval_evidence,
                       missing_device_trace_paths=missing,
                       expected_interval_count=samples,
                       parsed_interval_count=len(intervals),
                       diagnostic_raw_samples_us=[value for row in intervals
                                                  for value in row.get('durations_us', [])])
        if fatal:
            raise fatal[0]
        # Never let a missing trace conceal another schema, identity, or replay failure.
        if not all(row.get('clean') and row.get('samples') == 1 and row.get('total_device_rows') == 1
                   for row in intervals):
            raise MeasurementError('Every interval must contain exactly one target device kernel')
        if missing:
            raise IncompleteDeviceTrace('Missing device-kernel trace in %d/%d intervals; complete round rejected: %s'
                                        % (len(missing), samples, ', '.join(missing)))
        if not all(row.get('clean') and row.get('samples') == 1 and row.get('total_device_rows') == 1
                   for row in intervals):
            raise MeasurementError('Every interval must contain exactly one target device kernel')
        measured = profile_rows(directory, device_name)
        if (not measured.get('clean') or measured.get('samples') != samples
                or measured.get('total_device_rows') != samples):
            raise MeasurementError('Unexpected total number of target device-kernel samples')
        details.update(measured)
        return details
    finally:
        details.update(trace_paths=traces, replay_checks_this_measurement=storage.checks - checks_before)
        (directory / 'measurement.json').write_text(json.dumps(_safe(details), indent=2) + '\n')


def _summary(result):
    measured = [entry for entry in result['candidates'] if entry['status'] == 'measured']
    best = max(measured, key=lambda entry: entry['speedup'], default=None)
    invalid_replay = result['status'].startswith('baseline_replay_') or result['status'] == 'failed_state_restore'
    result['diagnostic_best_measured'] = (None if best is None else
                             {key: best[key] for key in ('axis', 'factor', 'speedup', 'regression',
                                                        'baseline_median_us', 'candidate_median_us')})
    result['best_merged'] = None
    result['recommendation'] = dict(factor=1, axis=-1, speedup=1.0, reason='candidate_independent_reference_pending')
    if invalid_replay:
        result['recommendation']['reason'] = 'no_valid_merged_recommendation_due_to_replay_failure'
    result['timing_evidence_usable'] = bool(measured) and not invalid_replay
    result['measured_configurations'] = len(measured)
    result['regressed_configurations'] = [dict(axis=e['axis'], factor=e['factor'], speedup=e['speedup'])
                                          for e in measured if e['regression']]
    result['all_measured_merged_regressed'] = bool(measured) and all(e['regression'] for e in measured)
    statuses = Counter(entry['status'] for entry in result['candidates'])
    result['configuration_status_counts'] = dict(sorted(statuses.items()))
    result['unsupported_configurations'] = [
        dict(axis=e['axis'], factor=e['factor'], stage=e['status'], reason=e.get('reason'))
        for e in result['candidates'] if e['status'] in ('unsupported_compile', 'unsupported_runtime')]
    result['runtime_fallback_configurations'] = [
        dict(axis=e['axis'], factor=e['factor'], reason=e.get('reason'))
        for e in result['candidates'] if e['status'] == 'runtime_fallback']
    pending_states = {'pending', 'compiling', 'measuring', 'partial'}
    failure_states = set(statuses) - pending_states - {
        'measured', 'unsupported_compile', 'unsupported_runtime', 'runtime_fallback', 'no_merge_opportunity', 'replay_unsupported'}
    result['failed_configurations'] = sum(statuses[s] for s in failure_states)
    result['search_complete'] = result['status'] == 'complete' and result['pending'] == 0
    result['best_scope'] = ('requested_search' if result['search_complete'] and not result['failed_configurations']
                            and not statuses['replay_unsupported'] else 'completed_configurations_only')
    if result['status'].startswith('baseline_replay_'):
        support = 'replay_not_validated'
    elif result['status'] == 'failed_state_restore':
        support = 'undetermined_due_to_failures'
    elif measured:
        support = 'supported_for_measured_configurations'
    elif statuses['replay_unsupported']:
        support = 'replay_not_validated'
    elif result['failed_configurations']:
        support = 'undetermined_due_to_failures'
    elif not result['search_complete']:
        support = 'incomplete'
    elif result['runtime_fallback_configurations']:
        support = 'runtime_fallback_in_requested_search'
    elif result['unsupported_configurations']:
        support = 'unsupported_in_requested_search'
    else:
        support = 'no_merge_opportunity'
    result['support_conclusion'] = support
    # A retained F=1 recommendation is an identity, not a measured speedup.
    result['recommendation']['baseline_identity_only'] = result['recommendation']['factor'] == 1


def sweep(kernel, raw, launch_args, storage, directory, *, torch, torch_npu, original_profile,
          profile_rows, raw_run_getter, checkpoint=None, axes=(0, 1, 2), factors=tuple(range(2, 65, 2)),
          rounds=3, samples=20, warmup=5, time_budget_s=0, compile_fn=None, option_type=None,
          support_checker=None, measure_fn=None):
    """Search one launch. Caller must disable recursive interception during this call.

    ``storage`` already holds pre-launch and original post-launch snapshots.
    ``raw_run_getter`` is the unpatched CompiledKernel.run.fget. A checkpoint is
    written before compilation and after every round/configuration, so a host
    timeout leaves the remaining search explicitly pending.
    """
    axes, factors = tuple(axes), tuple(factors)
    if os.environ.get('FLA_DUAL_SINGLE_ROUND')=='1':
        if os.environ.get('FLA_DUAL_SIDE')!='block': raise ValueError('dual one-round only valid for block side')
        rounds=1
    if (not axes or any(type(a) is not int or a not in (0, 1, 2) for a in axes)
            or len(set(axes)) != len(axes)):
        raise ValueError('axes must be unique members of 0,1,2')
    if (not factors or any(type(f) is not int or f < 2 or f > 64 or f % 2 for f in factors)
            or len(set(factors)) != len(factors)):
        raise ValueError('factors must be unique even integers from 2 through 64')
    from sampling_policy import validate as validate_sampling
    validate_sampling(rounds, samples, warmup, os.environ.get('FLA_DUAL_SINGLE_ROUND')=='1')
    if type(time_budget_s) not in (int, float) or not math.isfinite(time_budget_s) or time_budget_s < 0:
        raise ValueError('time_budget_s must be finite and nonnegative')
    if compile_fn is None:
        import triton
        compile_fn = triton.compile
    if option_type is None:
        from triton.backends.ascend.compiler import NPUOptions
        option_type = NPUOptions
    support_checker = support_checker or support_check
    measure_fn = measure_fn or measure_variant
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        original_grid = tuple(operator.index(value) for value in launch_args[:3])
    except TypeError as error:
        raise ValueError('Original grid must contain three nonnegative i32 integers') from error
    if (len(original_grid) != 3 or any(isinstance(value, bool) for value in launch_args[:3])
            or any(value < 0 or value > 2147483647 for value in original_grid)):
        raise ValueError('Original grid must contain three nonnegative i32 integers')
    original_ttir = kernel.asm.get('ttir')
    if not isinstance(original_ttir, str) or not original_ttir:
        raise ValueError('Original baseline TTIR is unavailable')
    options = compile_options(kernel, option_type)
    start = time.monotonic()
    candidates = [dict(axis=a, factor=f, status='pending', rounds=[]) for a in axes for f in factors]
    for entry in candidates:
        if 0 in original_grid or original_grid[entry['axis']] < 2:
            entry.update(status='no_merge_opportunity', reason=('EMPTY_GRID' if 0 in original_grid
                         else 'SELECTED_GRID_AXIS_HAS_LESS_THAN_TWO_PROGRAMS'))
    result = dict(schema_version=1, status='running', measurement_mode='guarded_state_restored_block_batching',
                  search_space=dict(axes=list(axes), factors=list(factors), configurations=len(candidates)),
                  rounds=rounds, samples_per_round=samples, warmup_per_round=warmup,
                  time_budget_s=time_budget_s, baseline_factor=1, original_grid=list(original_grid),
                  baseline_compiled_hash=kernel.hash, baseline_ttir_sha256=_sha(original_ttir),
                  compile_options=_safe(options), candidates=candidates,
                  correctness_scope='byte_exact_equivalence_to_original_launch; original_reference_checks_required',
                  performance_scope='device_kernel_duration; excludes_compile_guard_host_and_storage_copy_cost',
                  restore_original_post_state=False, best_merged=None, pending=len(candidates))

    def save():
        result['pending'] = sum(entry['status'] in ('pending', 'compiling', 'measuring', 'partial')
                                for entry in candidates)
        result['elapsed_s'] = time.monotonic() - start
        _summary(result)
        temporary = directory / 'tuning.json.tmp'
        temporary.write_text(json.dumps(_safe(result), indent=2) + '\n')
        os.replace(temporary, directory / 'tuning.json')
        if checkpoint is not None:
            checkpoint(result)

    def expired():
        return bool(time_budget_s and time.monotonic() - start >= time_budget_s)

    save()
    try:
        try:
            result['baseline_replay_support'] = support_checker(torch, kernel, raw, launch_args, {})
        except ReplayUnsupported as error:
            result.update(status='baseline_replay_unsupported', reason=str(error))
            return result
        except Exception as error:
            result.update(status='baseline_replay_error', reason=str(error), exception=traceback.format_exc())
            return result
        for entry in candidates:
            if entry['status'] == 'no_merge_opportunity':
                continue
            if expired():
                result.update(status='partial', reason='time_budget_exhausted')
                break
            entry['status'] = 'compiling'
            save()
            stage = 'compile'
            try:
                requested_options = dict(options, block_batching_factor=entry['factor'], block_batching_axis=entry['axis'])
                candidate = compile_fn(kernel.src, target=original_compilation_target(kernel), options=requested_options)
                report = getattr(candidate, 'block_batching', None)
                if not isinstance(report, dict):
                    raise RuntimeContractError('Compiler did not return a block batching bundle/report')
                entry['compile_report'] = _safe(report)
                bundle_ttir = candidate.baseline.asm.get('ttir')
                entry['bundle_baseline_ttir_sha256'] = _sha(bundle_ttir) if isinstance(bundle_ttir, str) else None
                if report.get('compile_status') == 'REJECTED':
                    entry.update(status='unsupported_compile', reason=report.get('reason', 'CANDIDATE_UNAVAILABLE'))
                    continue
                comparison = compare_ttir_structure(original_ttir, bundle_ttir,
                    directory/('ttir-axis%d-factor%d' % (entry['axis'], entry['factor'])))
                entry['baseline_ttir_comparison'] = comparison
                entry['baseline_ttir_identical'] = comparison['equivalent']
                if not entry['baseline_ttir_identical']:
                    entry.update(status='baseline_mismatch', reason='Independent baseline and bundle baseline TTIR differ')
                    continue
                if report.get('compile_status') != 'COMPILED_GUARDED' or candidate.candidate is None:
                    raise RuntimeContractError('Unexpected block batching compilation contract')
                # Accessing run initializes both native handles and can reject a
                # candidate whose load-time resource requirements are excessive.
                facade_run = candidate.run
                entry['compile_report'] = _safe(candidate.block_batching)
                if candidate.block_batching.get('compile_status') == 'REJECTED' or candidate.candidate is None:
                    entry.update(status='unsupported_compile', reason=candidate.block_batching.get('reason', 'LOAD_REJECTED'))
                    continue
                stage = 'support'
                entry['replay_support'] = {}
                for name, native in (('bundle_baseline', candidate.baseline), ('candidate', candidate.candidate)):
                    entry['replay_support'][name] = support_checker(torch, native, raw_run_getter(native), launch_args, {})
                candidate_name = candidate.candidate.packed_metadata['kernel_name']
                baseline_name = launch_args[5]['kernel_name']
                entry.update(status='measuring', candidate_compiled_hash=candidate.candidate.hash,
                             device_kernel_name=candidate_name, guarded_launches=0,
                             correctness='all_tensor_backing_storage_bytes_exact_each_replay')
                (directory / ('axis%d_factor%d' % (entry['axis'], entry['factor']))).mkdir(exist_ok=True)

                def candidate_launch():
                    check_launch_hooks(launch_args)
                    facade_run(*launch_args)
                    entry['last_launch'] = _safe(candidate.last_launch)
                    verify_applied(candidate.last_launch, entry['axis'], entry['factor'], original_grid)
                    entry['guarded_launches'] += 1

                stage = 'measure'
                for index in range(rounds):
                    if expired():
                        entry.update(status='partial', reason='time_budget_exhausted')
                        result.update(status='partial', reason='time_budget_exhausted')
                        break
                    order = ('baseline', 'candidate') if index % 2 == 0 else ('candidate', 'baseline')
                    round_result = dict(index=index, order=list(order))
                    entry['rounds'].append(round_result)
                    for name in order:
                        stage = 'measure_' + name
                        def baseline_launch():
                            check_launch_hooks(launch_args)
                            return raw(*launch_args)
                        launch = baseline_launch if name == 'baseline' else candidate_launch
                        variant_directory = directory / ('axis%d_factor%d' % (entry['axis'], entry['factor'])) / (
                            'round%d_%s' % (index + 1, name))
                        measured = measure_fn(storage, launch, variant_directory,
                                              baseline_name if name == 'baseline' else candidate_name,
                                              torch_npu=torch_npu, original_profile=original_profile,
                                              profile_rows=profile_rows, samples=samples, warmup=warmup)
                        values = measured.get('durations_us', ())
                        if (not preliminary_samples_valid(measured, samples) or len(values) != samples
                                or not all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in values)):
                            raise MeasurementError('Invalid or incomplete device sample result')
                        measured['median_us'] = statistics.median(values)
                        measured['mean_us'] = statistics.fmean(values)
                        measured['reported_statistic'] = 'arithmetic_mean'
                        round_result[name] = measured
                    round_result['speedup'] = round_result['baseline']['median_us'] / round_result['candidate']['median_us']
                    save()
                if entry['status'] != 'partial':
                    baseline_median = statistics.median(r['baseline']['median_us'] for r in entry['rounds'])
                    candidate_median = statistics.median(r['candidate']['median_us'] for r in entry['rounds'])
                    speedup = baseline_median / candidate_median
                    entry.update(status='measured', speedup=speedup, regression=speedup < 1,
                                 round_speedups=[r['speedup'] for r in entry['rounds']],
                                 regression_all_rounds=all(r['speedup'] < 1 for r in entry['rounds']),
                                 baseline_median_us=baseline_median, candidate_median_us=candidate_median,
                                 speedup_definition='median_baseline_round_medians_over_median_candidate_round_medians')
            except RestoreFailure:
                entry.update(status='failed_state_restore', exception=traceback.format_exc())
                result['status'] = 'failed_state_restore'
                raise
            except RuntimeFallback as error:
                entry.update(status='runtime_fallback', reason=str(error), last_launch=_safe(error.report))
            except ReplayUnsupported as error:
                entry.update(status='replay_unsupported', reason=str(error))
            except ReplayMismatch as error:
                entry.update(status='replay_mismatch', independent_reference_status='not_executed', failed_variant=stage.removeprefix('measure_'),
                             reason=str(error), exception=traceback.format_exc())
                if stage == 'measure_baseline':
                    # An unstable original replay cannot validate any candidate.
                    result.update(status='baseline_replay_mismatch', reason=str(error))
            except Exception as error:
                kind = ('runtime_contract_error' if isinstance(error, RuntimeContractError)
                        else 'compile_error' if stage == 'compile' else 'replay_support_error' if stage == 'support'
                        else 'measurement_error')
                entry.update(status=kind, failed_stage=stage, reason=str(error), exception=traceback.format_exc())
            finally:
                try:
                    storage.restore_post()
                    entry['restore_original_post_state'] = True
                except RestoreFailure:
                    entry.update(status='failed_state_restore', restore_original_post_state=False,
                                 exception=traceback.format_exc())
                    result['status'] = 'failed_state_restore'
                    raise
                finally:
                    entry['storage_checks_total'] = storage.checks
                    save()
            if result['status'] == 'baseline_replay_mismatch':
                break
        if result['status'] == 'running':
            result['status'] = 'complete'
    finally:
        try:
            storage.restore_post()
            result.update(restore_original_post_state=True, storage_state=storage.evidence(),
                          replay_checks_passed=storage.checks)
        except RestoreFailure:
            result['status'] = 'failed_state_restore'
            raise
        finally:
            save()
    return result
