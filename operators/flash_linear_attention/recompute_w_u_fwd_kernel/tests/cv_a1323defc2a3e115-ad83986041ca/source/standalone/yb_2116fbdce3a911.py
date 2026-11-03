
from opbench.fla_fixtures import load_fixture
import os
import csv
import glob
import time
import pytest
import torch
import torch_npu
from yb_5deca3efb6cf0b import recompute_w_u_fwd
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
kernel_name = 'recompute_w_u_fwd_kernel'

def get_precision_thresholds(dtype):
    return (0.008, 0.008)
B, T, H, D = (4, 2048, 8, 64)
BT = 64

@pytest.mark.functiontest
def test_recompute_w_u_fwd_kernel_function():
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/recompute_w_u_fwd_kernel/'
    k = load_fixture(os.path.join(load_dir, 'k.pt')).to(DEVICE)
    v = load_fixture(os.path.join(load_dir, 'v.pt')).to(DEVICE)
    beta = load_fixture(os.path.join(load_dir, 'beta.pt')).to(DEVICE)
    g_cumsum = load_fixture(os.path.join(load_dir, 'g_cumsum.pt')).to(DEVICE)
    A = load_fixture(os.path.join(load_dir, 'A.pt')).to(DEVICE)
    w, u = recompute_w_u_fwd(k, v, beta, g_cumsum, A, cu_seqlens=None)
    expected_w = load_fixture(os.path.join(load_dir, 'output_0.pt')).to(DEVICE)
    expected_u = load_fixture(os.path.join(load_dir, 'output_1.pt')).to(DEVICE)
    atol_w, rtol_w = get_precision_thresholds(w.dtype)
    torch.testing.assert_close(w, expected_w, atol=atol_w, rtol=rtol_w)
    print('Output 0 (w) precision check passed')
    atol_u, rtol_u = get_precision_thresholds(u.dtype)
    torch.testing.assert_close(u, expected_u, atol=atol_u, rtol=rtol_u)
    print('Output 1 (u) precision check passed')
    print(f'[{kernel_name}] Precision check completed successfully')

@pytest.mark.perftest
def test_recompute_w_u_fwd_kernel_performance():
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/recompute_w_u_fwd_kernel/'
    k = load_fixture(os.path.join(load_dir, 'k.pt')).to(DEVICE)
    v = load_fixture(os.path.join(load_dir, 'v.pt')).to(DEVICE)
    beta = load_fixture(os.path.join(load_dir, 'beta.pt')).to(DEVICE)
    g_cumsum = load_fixture(os.path.join(load_dir, 'g_cumsum.pt')).to(DEVICE)
    A = load_fixture(os.path.join(load_dir, 'A.pt')).to(DEVICE)
    result_dir = os.path.dirname(os.path.abspath(__file__))
    trace_dir = os.path.join(result_dir, f'prof_dir_{kernel_name}')
    with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.NPU], with_stack=False, record_shapes=False, profile_memory=False, schedule=torch_npu.profiler.schedule(wait=WAIT_ITERS, warmup=SCHEDULE_WARMUP, active=ACTIVE_ITERS, repeat=1, skip_first=SCHEDULE_SKIP_FIRST), experimental_config=experimental_config, on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(trace_dir)) as prof:
        for _ in range(TOTAL_ITERS):
            w, u = recompute_w_u_fwd(k, v, beta, g_cumsum, A, cu_seqlens=None)
            torch.npu.synchronize()
            prof.step()
        prof.stop()
    csv_path = find_op_statistic_csv(trace_dir)
    if csv_path is None:
        print(f'WARNING: op_statistic.csv not found in {trace_dir}')
        return
    profiler_results = parse_op_statistic(csv_path, kernel_filter=kernel_name)
    start_time = time.perf_counter()
    for _ in range(100):
        w, u = recompute_w_u_fwd(k, v, beta, g_cumsum, A, cu_seqlens=None)
        torch.npu.synchronize()
    elapsed = (time.perf_counter() - start_time) / 100 * 1000000.0
    shape_info = f'B={B}, T={T}, H={H}, D={D}, BT={BT}'
    if profiler_results:
        r = profiler_results[0]
        result_line = f"[{kernel_name}] {shape_info}, avg_time={r['avg_us']:.2f}us, min_time={r['min_us']:.2f}us, max_time={r['max_us']:.2f}us"
    else:
        result_line = f'[{kernel_name}] {shape_info}, avg_time={elapsed:.2f}us (perf_counter), profiler_result_not_found'
    print(result_line)
