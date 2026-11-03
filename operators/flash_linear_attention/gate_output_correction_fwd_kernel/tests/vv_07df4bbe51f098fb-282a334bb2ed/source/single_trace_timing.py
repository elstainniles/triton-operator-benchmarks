"""One profiler trace, individually identified target-kernel device durations.

No Torch import or device access occurs at import. Restoration/verification are
inside RECORD but outside target kernel execution; their time is not summed into
the target CSV Duration(us). Other device rows remain explicit excluded evidence.
"""
import csv
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics

from replay_state import AclStorageCopy, ReplayMismatch, ReplayUnsupported

PROTOCOL = 'single_trace_target_rows_v1'


class TraceContractError(ValueError):
    def __init__(self, message, evidence=None):
        super().__init__(message)
        self.evidence = evidence or {}


class MissingTargetRows(TraceContractError):
    """Only absent trace/target rows are eligible for one full-round retry."""


def require(condition, message, evidence=None):
    if not condition:
        raise TraceContractError(message, evidence)


def parse_trace(directory, target_name, expected_device, expected_samples):
    """Read and bind every selected row, while retaining every excluded row."""
    require(type(expected_samples) is int and expected_samples > 0, 'invalid expected sample count')
    require(isinstance(target_name, str) and target_name, 'missing exact target kernel name')
    require(str(expected_device).isdigit(), 'physical profiler device identity is not numeric')
    directory = Path(directory).resolve()
    files = sorted(directory.rglob('kernel_details.csv'))
    result = {'timing_protocol': PROTOCOL, 'target_kernel_name': target_name,
              'expected_profiler_device_id': str(expected_device), 'target_rows_validated': False,
              'window_clean_claimed': False, 'source_csv': [str(p) for p in files],
              'csv_sha256': {}, 'target_rows': [], 'excluded_rows': [],
              'excluded_rows_policy': 'retain_all_raw_rows_but_exclude_non_target_names_from_target_duration',
              'timing_scope': 'individual_target_kernel_csv_Duration(us)',
              'restoration_time_in_target_durations': False,
              'cpu_verification_time_in_target_durations': False,
              'profiler_export_time_in_target_durations': False}
    if not files:
        raise MissingTargetRows('Profiler produced no kernel_details.csv', result)
    require(len(files) == 1, 'single trace must have exactly one kernel_details.csv', result)
    path = files[0]
    raw = path.read_bytes()
    result['csv_sha256'][str(path)] = hashlib.sha256(raw).hexdigest()
    reader = csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))
    columns = reader.fieldnames or []
    required = {'Name', 'Device_id', 'Stream ID', 'Task ID', 'Start Time(us)', 'Duration(us)'}
    require(required.issubset(columns) and len(columns) == len(set(columns)),
            'missing/duplicate profiler identity or timing columns', result)
    selected = []
    all_rows = []
    for index, row in enumerate(reader, start=2):
        require(None not in row and all(value is not None for value in row.values()),
                'malformed profiler CSV row', result)
        item = {'source_csv': str(path), 'csv_row_number': index, 'raw': dict(row)}
        all_rows.append(item)
        if row['Name'] != target_name:
            result['excluded_rows'].append(item)
            continue
        result['target_rows'].append(item)
        require(row['Device_id'] == str(expected_device), 'target row belongs to wrong physical device', result)
        require(row['Stream ID'].isdigit() and row['Task ID'].isdigit(), 'target Stream ID/Task ID missing or invalid', result)
        try:
            start = Decimal(row['Start Time(us)'].strip())
            duration_decimal = Decimal(row['Duration(us)'].strip())
            duration = float(duration_decimal)
        except (ValueError, InvalidOperation):
            raise TraceContractError('target row time is not numeric', result)
        require(start.is_finite() and start >= 0 and duration_decimal.is_finite()
                and math.isfinite(duration) and duration > 0, 'target row duration/start is not finite positive', result)
        item['identity'] = {'name': row['Name'], 'device_id': row['Device_id'],
                            'stream_id': row['Stream ID'], 'task_id': row['Task ID'],
                            'start_time_us': row['Start Time(us)'], 'duration_us': row['Duration(us)']}
        selected.append((start, duration_decimal, duration, item))
    result['total_device_rows'] = len(all_rows)
    result['target_device_rows'] = len(selected)
    result['excluded_device_rows'] = len(result['excluded_rows'])
    result['other_device_kernels'] = {}
    for item in result['excluded_rows']:
        name = item['raw']['Name']
        result['other_device_kernels'][name] = result['other_device_kernels'].get(name, 0) + 1
    result['all_device_ids'] = sorted({item['raw']['Device_id'] for item in all_rows})
    streams = {int(item['identity']['stream_id']) for _, _, _, item in selected}
    tasks = [int(item['identity']['task_id']) for _, _, _, item in selected]
    require(len(streams) <= 1, 'target launches used multiple streams', result)
    require(len(tasks) == len(set(tasks)), 'duplicate target Task ID in single trace', result)
    selected.sort(key=lambda value: (value[0], value[3]['csv_row_number']))
    for prior, current in zip(selected, selected[1:]):
        require(current[0] >= prior[0] + prior[1], 'target kernel rows overlap despite serial synchronization', result)
    result['target_rows'] = [item for _, _, _, item in selected]
    result['durations_us'] = [duration for _, _, duration, _ in selected]
    result['samples'] = len(selected)
    result['device_ids'] = [str(expected_device)] if selected else []
    result['stream_ids'] = [str(stream) for stream in sorted(streams)]
    result['target_task_ids'] = [item['identity']['task_id'] for _, _, _, item in selected]
    if len(selected) < expected_samples:
        raise MissingTargetRows('Missing target rows: expected %d, found %d; entire round rejected' %
                                (expected_samples, len(selected)), result)
    require(len(selected) == expected_samples, 'too many target rows in one trace', result)
    values = result['durations_us']
    result.update(target_rows_validated=True, mean_us=statistics.mean(values), median_us=statistics.median(values),
                  min_us=min(values), max_us=max(values))
    return result


def _require_acl_storage(storage):
    if type(storage.acl) is not AclStorageCopy or not storage.storages:
        raise ReplayUnsupported('single-trace timing requires actual AclStorageCopy NPU storage restoration')
    if any(entry.get('npu_format') is None or not str(entry.get('device', '')).startswith(('npu:', 'privateuseone:'))
           for entry in storage.storages):
        raise ReplayUnsupported('single-trace timing supports only raw ACL-restored NPU backing storages')


def restored_replays(storage, launch, profiler_factory, details, *, samples, warmup):
    """Full restore and byte-exact verify for each replay; one RECORD span."""
    _require_acl_storage(storage)
    checks_before = storage.checks
    storage_bytes = sum(entry['bytes'] for entry in storage.storages)
    storage_indices = list(range(len(storage.storages)))
    details.update(timing_protocol=PROTOCOL, original_calls=1, warmup=warmup,
                   warmup_completed=0, samples_completed=0, samples_attempted=0,
                   restoration_outside_capture=False, restoration_outside_target_timing=True,
                   verification_outside_target_timing=True, restore_original_post_state=False,
                   preserves_aliases_strides_offsets_addresses=False,
                   storage_transfer_method='AclStorageCopy.aclrtMemcpy', restore_pre_count=0,
                   storage_checks=[], checks_before_measurement=checks_before)

    def replay(phase, index):
        before = storage.checks
        storage.restore_pre()
        details['restore_pre_count'] += 1
        if phase == 'sample':
            details['samples_attempted'] += 1
        launch()
        storage.synchronize()
        storage.verify_post()
        if storage.checks != before + 1:
            raise ReplayMismatch('expected exactly one complete storage check per target launch')
        details['storage_checks'].append({'phase': phase, 'index': index,
            'checks_before': before, 'checks_after': storage.checks,
            'restore_pre_completed': True, 'serial_synchronize_after_launch': True,
            'full_storage_post_state_verified': True, 'checked_storage_indices': storage_indices,
            'checked_bytes_total': storage_bytes,
            'capture_phase': 'outside_capture' if phase == 'warmup' else 'inside_record',
            'restore_stage': 'before_target_launch', 'verify_stage': 'after_target_synchronize'})
        details['warmup_completed' if phase == 'warmup' else 'samples_completed'] += 1

    def action(profiler, expected):
        observed = getattr(getattr(profiler, 'current_action', None), 'name', None)
        if observed != expected or getattr(profiler, 'stopped', False):
            raise ReplayMismatch('Expected single-trace profiler action %s, got %s' % (expected, observed))

    try:
        for index in range(warmup):
            replay('warmup', index)
        with profiler_factory() as profiler:
            for index in range(samples):
                action(profiler, 'RECORD_AND_SAVE' if index == samples - 1 else 'RECORD')
                replay('sample', index)
                profiler.step()
            action(profiler, 'NONE')
        details.update(preserves_aliases_strides_offsets_addresses=True,
                       replay_equivalence='all_tensor_backing_storage_bytes_exact_each_replay')
    finally:
        details['replay_checks_passed'] = storage.checks
        details['replay_checks_this_measurement'] = storage.checks - checks_before
        storage.restore_post()
        details['restore_original_post_state'] = True
        details['storage_state'] = storage.evidence()


def measure_once(storage, launch, directory, device_name, *, torch_npu, original_profile,
                 profile_rows, samples, warmup):
    """Collect all target samples in one trace and export after state restoration."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    traces = []
    details = {'timing_protocol': PROTOCOL, 'window_clean_claimed': False}
    visible = os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '')
    expected = visible if os.environ.get('FLA_BLOCK_EXECUTION_MODE') == 'local' else '0'
    require(expected.isdigit(), 'single-trace timing requires one verified physical profiler device')

    def profiler_factory():
        handler = torch_npu.profiler.tensorboard_trace_handler(str(directory), analyse_flag=False)

        def trace_ready(profiler):
            path = Path(profiler.prof_if.prof_path).resolve()
            require(directory.resolve() in path.parents, 'profiler trace escaped measurement directory')
            traces.append(str(path))
            handler(profiler)

        config = torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization, l2_cache=False)
        return original_profile(activities=[torch_npu.profiler.ProfilerActivity.NPU], record_shapes=True,
            profile_memory=False, with_stack=False, experimental_config=config,
            schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=samples, repeat=1),
            on_trace_ready=trace_ready)

    try:
        restored_replays(storage, launch, profiler_factory, details, samples=samples, warmup=warmup)
        if not traces:
            raise MissingTargetRows('single-trace profiler emitted no trace callback')
        require(len(traces) == 1, 'single-trace profiler emitted multiple trace callbacks')
        analyse = getattr(torch_npu.profiler, 'analyse', None)
        if analyse is None:
            from torch_npu.profiler.profiler import analyse
        analyse(str(directory), max_process_number=2)
        details.update(parse_trace(traces[0], device_name, expected, samples))
        return details
    except TraceContractError as error:
        details.update(error.evidence)
        details.update(timing_error_type=type(error).__name__, timing_error=str(error))
        raise
    finally:
        details['trace_paths'] = traces
        (directory / 'measurement.json').write_text(json.dumps(details, indent=2) + '\n')


def preliminary_samples_valid(row, count):
    """Producer validation only; the gate separately re-reads raw CSV evidence."""
    if row.get('timing_protocol') != PROTOCOL:
        return row.get('clean') is True and row.get('samples') == count and row.get('total_device_rows') == count
    values = row.get('durations_us')
    return (row.get('target_rows_validated') is True and row.get('window_clean_claimed') is False
            and row.get('samples') == row.get('target_device_rows') == count
            and isinstance(values, list) and len(values) == count
            and all(type(value) in (int, float) and math.isfinite(value) and value > 0 for value in values))


def check_samples(row, count, warmup):
    """Strict new-protocol gate; every duration is re-derived from preserved CSV."""
    require(isinstance(row, dict) and row.get('timing_protocol') == PROTOCOL, 'wrong single-trace timing protocol')
    require(row.get('clean') is not True and row.get('window_clean_claimed') is False,
            'single-trace timing must not claim old clean-window isolation')
    require(row.get('warmup_completed') == warmup and row.get('samples_completed') == count
            and row.get('samples_attempted') == count, 'incomplete single-trace warmup/sampling')
    require(row.get('restoration_outside_capture') is False and row.get('restoration_outside_target_timing') is True
            and row.get('verification_outside_target_timing') is True, 'restoration timing scope is incorrect')
    require(row.get('preserves_aliases_strides_offsets_addresses') is True
            and row.get('restore_original_post_state') is True
            and row.get('replay_equivalence') == 'all_tensor_backing_storage_bytes_exact_each_replay'
            and row.get('storage_transfer_method') == 'AclStorageCopy.aclrtMemcpy', 'full ACL replay proof missing')
    require(row.get('replay_checks_this_measurement') == count + warmup
            and row.get('restore_pre_count') == count + warmup, 'complete per-replay restore/check count missing')
    storage = row.get('storage_state', {})
    storages = storage.get('storages', [])
    require(storages and storage.get('argument_layouts'), 'full storage layout/digest evidence missing')
    for entry in storages:
        require(entry.get('pre_sha256') and entry.get('post_sha256') and entry.get('npu_format') is not None
                and str(entry.get('device', '')).startswith(('npu:', 'privateuseone:')),
                'storage is not fully snapshotted raw NPU memory')
    byte_count = sum(entry['bytes'] for entry in storages)
    require(storage.get('total_bytes') == byte_count, 'storage byte count differs')
    checks = row.get('storage_checks')
    start = row.get('checks_before_measurement')
    require(type(start) is int and isinstance(checks, list) and len(checks) == count + warmup,
            'per-replay storage check log missing')
    for index, record in enumerate(checks):
        phase = 'warmup' if index < warmup else 'sample'
        phase_index = index if index < warmup else index - warmup
        require(record.get('phase') == phase and record.get('index') == phase_index
                and record.get('checks_before') == start + index and record.get('checks_after') == start + index + 1,
                'storage check sequence differs')
        require(record.get('restore_pre_completed') is True and record.get('serial_synchronize_after_launch') is True
                and record.get('full_storage_post_state_verified') is True
                and record.get('checked_storage_indices') == list(range(len(storages)))
                and record.get('checked_bytes_total') == byte_count
                and record.get('restore_stage') == 'before_target_launch'
                and record.get('verify_stage') == 'after_target_synchronize'
                and record.get('capture_phase') == ('outside_capture' if phase == 'warmup' else 'inside_record'),
                'per-replay storage restoration/verification scope differs')
    require(row.get('replay_checks_passed') == start + count + warmup, 'storage final check counter differs')
    traces = row.get('trace_paths')
    require(isinstance(traces, list) and len(traces) == 1, 'exactly one profiler trace required')
    parsed = parse_trace(traces[0], row.get('target_kernel_name'), row.get('expected_profiler_device_id'), count)
    for key in ('target_kernel_name', 'expected_profiler_device_id', 'target_rows_validated', 'source_csv', 'csv_sha256',
                'target_rows', 'excluded_rows', 'excluded_rows_policy', 'timing_scope', 'restoration_time_in_target_durations',
                'cpu_verification_time_in_target_durations', 'profiler_export_time_in_target_durations', 'total_device_rows',
                'target_device_rows', 'excluded_device_rows', 'other_device_kernels', 'all_device_ids', 'durations_us',
                'samples', 'device_ids', 'stream_ids', 'target_task_ids'):
        require(row.get(key) == parsed[key], 'raw single-trace CSV/evidence mismatch: ' + key)
    return statistics.median(parsed['durations_us'])
