"""Run one imported suite in an audited isolated runtime, with graph optimization off.

Preserves upstream profiling; deliberately does not publish accepted performance data.
"""
import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import os
from pathlib import Path
import runpy
import shutil
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opbench.catalog import read_json, resolve
from opbench.device import Monitor, identity
from opbench.environment import audit, configure, digest
from opbench.graph_tests import verify_graph_suite
from opbench.runner import save


def validate_request(metadata, input_pt, case_ids):
    if metadata['input_mode'] == 'pt_replay':
        if input_pt is None or not input_pt.is_file():
            raise ValueError('Missing upstream input .pt; provide --input-pt from a trusted source')
    elif input_pt is not None:
        raise ValueError('--input-pt applies only to TG01/TG02/TG03')
    if case_ids and metadata['input_mode'] != 'generated_outputs_only':
        raise ValueError('--case-id applies only to TG07 (MoE)')
    if case_ids and any(i < 0 or i >= metadata['case_count'] for i in case_ids):
        raise ValueError('MoE case IDs must be 0..99')


def execute_script(metadata, staging, input_pt, case_ids):
    """Caller already owns the card lock and audited runtime; never used by host CI."""
    entrypoint = staging / metadata['entrypoint']
    old_path, old_cwd, old_argv = list(sys.path), Path.cwd(), list(sys.argv)
    try:
        os.chdir(staging)
        # Only the copied test helpers are added. TA/Torch still come from the audited runtime.
        sys.path.insert(0, str(staging))
        if metadata['input_mode'] == 'pt_replay':
            scope = runpy.run_path(str(entrypoint), run_name='graph_test_replay')
            scope['test_get_last_loc_kernel'](str(input_pt))
            scope['test_perf'](str(input_pt))
        else:
            sys.argv = [str(entrypoint)]
            if case_ids:
                sys.argv += ['--cases', *map(str, case_ids)]
            try:
                runpy.run_path(str(entrypoint), run_name='__main__')
            except SystemExit as exc:
                if exc.code not in (None, 0):
                    raise RuntimeError(f'Upstream test exited with code {exc.code}') from exc
    finally:
        sys.path[:] = old_path
        sys.argv[:] = old_argv
        os.chdir(old_cwd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operator', required=True)
    parser.add_argument('--physical-card', type=int, choices=range(8), required=True)
    parser.add_argument('--environment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--input-pt', type=Path)
    parser.add_argument('--case-id', type=int, action='append', default=[])
    args = parser.parse_args()
    entry, path = resolve(args.operator)
    if entry.get('runner') != 'graph_tests':
        parser.error('Choose a TG01..TG07 operator; use tools/run.py for FlagGems')
    metadata = read_json(path / 'operator.json')
    input_pt = args.input_pt.resolve() if args.input_pt is not None else None
    try:
        validate_request(metadata, input_pt, args.case_id)
        source_audit = verify_graph_suite(path)
    except ValueError as exc:
        parser.error(str(exc))
    expected = read_json(args.environment)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = dict(operator=entry, completed=False, status='pending',
                  source_revision=metadata['source']['revision'], source_audit=source_audit,
                  compile_options=metadata['compile_options'], physical_card=args.physical_card,
                  selected_case_ids=args.case_id or None,
                  numerical_status='NOT_RUN', performance_status='NO_ACCEPTED_MEASUREMENT',
                  started_utc=datetime.now(timezone.utc).isoformat())
    result_path = output / 'result.json'
    save(result_path, result)
    try:
        configure(expected, output, args.physical_card)
        import fcntl
        lockroot = Path('/tmp') / f'triton-operator-benchmarks-{os.getuid()}'
        lockroot.mkdir(mode=0o700, exist_ok=True)
        if lockroot.is_symlink() or lockroot.stat().st_uid != os.getuid():
            raise RuntimeError('Invalid per-user card lock directory')
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
                    raise RuntimeError('Expected one explicitly selected NPU')
                torch.npu.set_device(0)
                backend = AscendBackend(triton.runtime.driver.active.get_current_target())
                options = backend.parse_options({'enable_graph_optimize': False})
                if getattr(options, 'enable_graph_optimize', None) is not False:
                    raise RuntimeError('This backend does not confirm enable_graph_optimize=False')
                result['backend_graph_option_verified'] = True
                staging = output / 'work'
                staging.mkdir()
                for record in metadata['files']:
                    shutil.copyfile(path / record['executable'], staging / record['executable'])
                if input_pt is not None:
                    result['input_pt_sha256'] = digest(input_pt)
                # Pin the imported suites' supported local outputs/knobs, not inherited values.
                if entry['id'] in {'TG04', 'TG05', 'TG06'}:
                    os.environ['TRITON_ALL_BLOCKS_PARALLEL'] = '1'
                if entry['id'] == 'TG07':
                    os.environ['MOE_PROF_ROOT'] = str(staging / 'prof_fused_moe_dir')
                    os.environ['MOE_PROF_DIR'] = str(staging / 'prof_fused_moe_dir')
                    os.environ['MOE_AUTOTUNE_RESET'] = '1'
                save(result_path, result)
                monitor.require_clear()
                with (output / 'test.log').open('w', encoding='utf-8', buffering=1) as log:
                    with redirect_stdout(log), redirect_stderr(log):
                        execute_script(metadata, staging, input_pt, args.case_id)
                monitor.require_clear()
                result['status'] = 'upstream_script_finished'
                result['numerical_status'] = ('OUTPUT_SAVED_REFERENCE_MISSING'
                    if metadata['input_mode'] == 'generated_outputs_only' else 'UPSTREAM_CHECKS_PASSED')
            result['monitor_clear'] = monitor.eligible_since(0)
            result['performance_status'] = ('RAW_PROFILE_REQUIRES_REVIEW' if result['monitor_clear']
                                            else 'REJECTED_INTERFERENCE')
            result['completed'] = True
    except BaseException:
        result['status'] = 'failed'
        result['error'] = traceback.format_exc()
        raise
    finally:
        result['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save(result_path, result)
    print(str(result_path))


if __name__ == '__main__':
    main()
