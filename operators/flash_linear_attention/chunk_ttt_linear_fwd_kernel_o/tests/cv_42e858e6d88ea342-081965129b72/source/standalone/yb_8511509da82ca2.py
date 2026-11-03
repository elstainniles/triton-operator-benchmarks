
from opbench.fla_fixtures import load_fixture
import os
import csv
import glob
import pytest
import torch
import torch_npu
from yb_7bff0dd87eff27 import chunk_ttt_linear_fwd_h
from yb_7bff0dd87eff27 import chunk_ttt_linear_fwd_o
NPU_ID = os.environ.get('BENCH_NPU_ID', '0')
torch.npu.set_device(int(NPU_ID))
DEVICE = f'npu:{torch.npu.current_device()}'
ACTIVE_ITERS = 20
WAIT_ITERS = 1
SCHEDULE_WARMUP = 1
SCHEDULE_SKIP_FIRST = 1
TOTAL_ITERS = SCHEDULE_SKIP_FIRST + WAIT_ITERS + SCHEDULE_WARMUP + ACTIVE_ITERS
experimental_config = torch_npu.profiler._ExperimentalConfig(aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization, profiler_level=torch_npu.profiler.ProfilerLevel.Level1, l2_cache=False)
kernel_name = 'chunk_ttt_linear_fwd_kernel_o'
B, T, H, D = (4, 2048, 8, 64)
chunk_size = 16
scale = D ** (-0.5)

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
def test_chunk_ttt_linear_fwd_kernel_o_function():
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_ttt_linear_fwd_kernel_o/'
    q = load_fixture(os.path.join(load_dir, 'q.pt')).to(DEVICE)
    k = load_fixture(os.path.join(load_dir, 'k.pt')).to(DEVICE)
    v_new = load_fixture(os.path.join(load_dir, 'v_new.pt')).to(DEVICE)
    eta = load_fixture(os.path.join(load_dir, 'eta.pt')).to(DEVICE)
    h = load_fixture(os.path.join(load_dir, 'h.pt')).to(DEVICE)
    hb = load_fixture(os.path.join(load_dir, 'hb.pt')).to(DEVICE)
    output = chunk_ttt_linear_fwd_o(q, k, v_new, eta, h, hb, scale, None, chunk_size)
    if isinstance(output, tuple):
        for i, out in enumerate(output):
            saved_output = load_fixture(os.path.join(load_dir, f'output_{i}.pt')).to(DEVICE)
            atol, rtol = get_precision_thresholds(out.dtype)
            torch.testing.assert_close(out, saved_output, atol=atol, rtol=rtol)
            print(f'[{kernel_name}] output_{i} 精度验证通过')
    else:
        saved_output = load_fixture(os.path.join(load_dir, 'output.pt')).to(DEVICE)
        atol, rtol = get_precision_thresholds(output.dtype)
        torch.testing.assert_close(output, saved_output, atol=atol, rtol=rtol)
        print(f'[{kernel_name}] output 精度验证通过')

@pytest.mark.perftest
def test_chunk_ttt_linear_fwd_kernel_o_performance():
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_ttt_linear_fwd_kernel_o/'
    q = load_fixture(os.path.join(load_dir, 'q.pt')).to(DEVICE)
    k = load_fixture(os.path.join(load_dir, 'k.pt')).to(DEVICE)
    v_new = load_fixture(os.path.join(load_dir, 'v_new.pt')).to(DEVICE)
    eta = load_fixture(os.path.join(load_dir, 'eta.pt')).to(DEVICE)
    h = load_fixture(os.path.join(load_dir, 'h.pt')).to(DEVICE)
    hb = load_fixture(os.path.join(load_dir, 'hb.pt')).to(DEVICE)
    result_dir = os.path.dirname(os.path.abspath(__file__))
    trace_dir = os.path.join(result_dir, f'prof_dir_{kernel_name}')
    with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.NPU], with_stack=False, record_shapes=False, profile_memory=False, schedule=torch_npu.profiler.schedule(wait=WAIT_ITERS, warmup=SCHEDULE_WARMUP, active=ACTIVE_ITERS, repeat=1, skip_first=SCHEDULE_SKIP_FIRST), experimental_config=experimental_config, on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(trace_dir)) as prof:
        for _ in range(TOTAL_ITERS):
            output = chunk_ttt_linear_fwd_o(q, k, v_new, eta, h, hb, scale, None, chunk_size)
            torch.npu.synchronize()
            prof.step()
        prof.stop()
    csv_path = find_op_statistic_csv(trace_dir)
    if csv_path is None:
        print(f'WARNING: op_statistic.csv not found in {trace_dir}')
        return
    results = parse_op_statistic(csv_path, kernel_filter=kernel_name)
    if not results:
        print(f'WARNING: kernel {kernel_name} not found in op_statistic.csv')
        return
    r = results[0]
    shape_info = f'B={B}, T={T}, H={H}, D={D}'
    print(f"[{kernel_name}] shape={shape_info}, avg_time={r['avg_us']:.2f}us, min_time={r['min_us']:.2f}us, max_time={r['max_us']:.2f}us")
