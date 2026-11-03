
from opbench.fla_fixtures import load_fixture
import os
import csv
import glob
import pytest
import torch
import torch_npu
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
kernel_name = 'chunk_oja_bwd_kernel_dA'

def get_precision_thresholds(dtype):
    return (0.008, 0.008)

@pytest.mark.functiontest
def test_chunk_oja_bwd_kernel_dA_function():
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_oja_bwd_kernel_dA/'
    from yb_8b1f9f64c8c620 import chunk_oja_bwd_dA
    v = load_fixture(os.path.join(load_dir, 'v.pt')).to(DEVICE)
    gv = load_fixture(os.path.join(load_dir, 'gv.pt')).to(DEVICE)
    do = load_fixture(os.path.join(load_dir, 'do.pt')).to(DEVICE)
    scale = load_fixture(os.path.join(load_dir, 'scale.pt')).item()
    block_size = load_fixture(os.path.join(load_dir, 'block_size.pt')).item()
    output = chunk_oja_bwd_dA(v, gv, do, scale, None, None, block_size)
    if isinstance(output, tuple):
        for i, out in enumerate(output):
            expected_out = load_fixture(os.path.join(load_dir, f'output_{i}.pt')).to(DEVICE)
            atol, rtol = get_precision_thresholds(out.dtype)
            torch.testing.assert_close(out, expected_out, atol=atol, rtol=rtol)
            print(f'Output {i} precision check passed')
    else:
        expected_output = load_fixture(os.path.join(load_dir, 'output.pt')).to(DEVICE)
        atol, rtol = get_precision_thresholds(output.dtype)
        torch.testing.assert_close(output, expected_output, atol=atol, rtol=rtol)
        print('Output precision check passed')
    print(f'[{kernel_name}] Precision check completed successfully')

@pytest.mark.perftest
def test_chunk_oja_bwd_kernel_dA_performance():
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_oja_bwd_kernel_dA/'
    from yb_8b1f9f64c8c620 import chunk_oja_bwd_dA
    v = load_fixture(os.path.join(load_dir, 'v.pt')).to(DEVICE)
    gv = load_fixture(os.path.join(load_dir, 'gv.pt')).to(DEVICE)
    do = load_fixture(os.path.join(load_dir, 'do.pt')).to(DEVICE)
    scale = load_fixture(os.path.join(load_dir, 'scale.pt')).item()
    block_size = load_fixture(os.path.join(load_dir, 'block_size.pt')).item()
    result_dir = os.path.dirname(os.path.abspath(__file__))
    trace_dir = os.path.join(result_dir, f'prof_dir_{kernel_name}')

    def run_kernel():
        return chunk_oja_bwd_dA(v, gv, do, scale, None, None, block_size)
    with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.NPU], with_stack=False, record_shapes=False, profile_memory=False, schedule=torch_npu.profiler.schedule(wait=WAIT_ITERS, warmup=SCHEDULE_WARMUP, active=ACTIVE_ITERS, repeat=1, skip_first=SCHEDULE_SKIP_FIRST), experimental_config=experimental_config, on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(trace_dir)) as prof:
        for _ in range(TOTAL_ITERS):
            output = run_kernel()
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
    for r in results:
        print(f"[{kernel_name}] B=4, T=4096, H=4, D=128, avg_time={r['avg_us']:.2f}us, min_time={r['min_us']:.2f}us, max_time={r['max_us']:.2f}us")
