
from opbench.fla_fixtures import load_fixture
import os
import csv
import glob
import time
import pytest
import torch
import torch_npu
from yb_35b249daa9eeeb import solve_tril
NPU_ID = os.environ.get('BENCH_NPU_ID', '0')
torch.npu.set_device(int(NPU_ID))
DEVICE = f'npu:{torch.npu.current_device()}'
ACTIVE_ITERS = 20
WAIT_ITERS = 1
SCHEDULE_WARMUP = 1
SCHEDULE_SKIP_FIRST = 1
TOTAL_ITERS = SCHEDULE_SKIP_FIRST + WAIT_ITERS + SCHEDULE_WARMUP + ACTIVE_ITERS
experimental_config = torch_npu.profiler._ExperimentalConfig(aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization, profiler_level=torch_npu.profiler.ProfilerLevel.Level1, l2_cache=False)

def find_op_statistic_csv(prof_dir):
    pattern = os.path.join(prof_dir, '**', 'ASCEND_PROFILER_OUTPUT', 'op_statistic.csv')
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        return None
    matches.sort(key=os.path.getmtime, reverse=True)
    return matches[0]

def parse_op_statistic(csv_path, kernel_filter=None):
    results = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            op_type = row.get('OP Type', '')
            if kernel_filter and kernel_filter not in op_type:
                continue
            results.append({'op_type': op_type, 'avg_us': float(row.get('Avg Time(us)', 0)), 'max_us': float(row.get('Max Time(us)', 0)), 'min_us': float(row.get('Min Time(us)', 0)), 'count': int(row.get('Count', 0))})
    return results

def get_precision_thresholds(dtype):
    if dtype == torch.float32:
        return (1e-05, 1e-05)
    elif dtype == torch.float16:
        return (0.001, 0.001)
    elif dtype == torch.bfloat16:
        return (0.005, 0.005)
    else:
        return (0.001, 0.001)

@pytest.mark.functiontest
def test_merge_16x16_to_64x64_inverse_kernel_function():
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/merge_16x16_to_64x64_inverse_kernel/'
    A_input = load_fixture(os.path.join(load_dir, 'input_A_input.pt')).to(DEVICE)
    output = solve_tril(A_input)
    torch.npu.synchronize()
    expected_output = load_fixture(os.path.join(load_dir, 'output.pt')).to(DEVICE)
    atol, rtol = get_precision_thresholds(output.dtype)
    torch.testing.assert_close(output, expected_output, atol=atol, rtol=rtol)
    print(f'[merge_16x16_to_64x64_inverse_kernel] Precision check passed!')

@pytest.mark.perftest
def test_merge_16x16_to_64x64_inverse_kernel_performance():
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/merge_16x16_to_64x64_inverse_kernel/'
    A_input = load_fixture(os.path.join(load_dir, 'input_A_input.pt')).to(DEVICE)
    result_dir = os.path.dirname(os.path.abspath(__file__))
    trace_dir = os.path.join(result_dir, 'prof_dir_merge_16x16_to_64x64_inverse_kernel')

    def run_kernel():
        return solve_tril(A_input)
    with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.NPU], with_stack=False, record_shapes=False, profile_memory=False, schedule=torch_npu.profiler.schedule(wait=WAIT_ITERS, warmup=SCHEDULE_WARMUP, active=ACTIVE_ITERS, repeat=1, skip_first=SCHEDULE_SKIP_FIRST), experimental_config=experimental_config, on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(trace_dir)) as prof:
        for _ in range(TOTAL_ITERS):
            output = run_kernel()
            torch.npu.synchronize()
            prof.step()
        prof.stop()
    perf_iters = 50
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(perf_iters):
        output = run_kernel()
    torch.npu.synchronize()
    t1 = time.perf_counter()
    perf_avg_us = (t1 - t0) / perf_iters * 1000000.0
    csv_path = find_op_statistic_csv(trace_dir)
    if csv_path is None:
        print(f'WARNING: op_statistic.csv not found in {trace_dir}')
        return
    results = parse_op_statistic(csv_path, kernel_filter='merge_16x16_to_64x64_inverse_kernel')
    if not results:
        all_results = parse_op_statistic(csv_path)
        print(f'No matching kernel found. All ops:')
        for r in all_results[:20]:
            print(f"  {r['op_type']}: avg={r['avg_us']:.2f}us")
        return
    r = results[0]
    B, T, H, BT = A_input.shape
    shape_info = f'B={B}, T={T}, H={H}, BT={BT}'
    result_line = f"[merge_16x16_to_64x64_inverse_kernel] shape={shape_info}, avg_time={r['avg_us']:.2f}us, min_time={r['min_us']:.2f}us, max_time={r['max_us']:.2f}us"
    print(result_line)
    print(f'perf_counter avg: {perf_avg_us:.2f}us')
