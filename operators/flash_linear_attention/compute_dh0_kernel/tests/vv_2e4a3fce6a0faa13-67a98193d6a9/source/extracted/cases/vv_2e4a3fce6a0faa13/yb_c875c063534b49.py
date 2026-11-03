
from opbench.fla_fixtures import load_fixture
import os
import csv
import glob
import time
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
RESULT_DIR = os.path.dirname(os.path.abspath(__file__))
experimental_config = torch_npu.profiler._ExperimentalConfig(aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization, profiler_level=torch_npu.profiler.ProfilerLevel.Level1, l2_cache=False)
kernel_name = 'compute_dh0_kernel'

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

@pytest.mark.perftest
def test_compute_dh0_kernel_perftest(load_dir):
    from yb_717c6a81dac3a7 import compute_dh0_triton
    B, T, D, W = (4, 1024, 4096, 4)
    inputs = load_fixture(os.path.join(load_dir, f'{kernel_name}_in.pt'))
    dy = inputs['dy'].to(DEVICE)
    y = inputs['y'].to(DEVICE)
    weight = inputs['weight'].to(DEVICE)
    initial_state = inputs['initial_state'].to(DEVICE)
    activation = inputs['activation']
    seq_idx = inputs['seq_idx']
    trace_dir = os.path.join(RESULT_DIR, f'prof_dir_{kernel_name}')
    with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.NPU], with_stack=False, record_shapes=False, profile_memory=False, schedule=torch_npu.profiler.schedule(wait=WAIT_ITERS, warmup=SCHEDULE_WARMUP, active=ACTIVE_ITERS, repeat=1, skip_first=SCHEDULE_SKIP_FIRST), experimental_config=experimental_config, on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(trace_dir)) as prof:
        for _ in range(TOTAL_ITERS):
            dh0 = compute_dh0_triton(dy, y, weight, initial_state, activation, seq_idx)
            torch.npu.synchronize()
            prof.step()
        prof.stop()
    csv_path = find_op_statistic_csv(trace_dir)
    if csv_path is None:
        print(f'WARNING: op_statistic.csv not found in {trace_dir}')
        return
    results = parse_op_statistic(csv_path, kernel_filter='compute_dh0_kernel')
    start = time.perf_counter()
    for _ in range(100):
        dh0 = compute_dh0_triton(dy, y, weight, initial_state, activation, seq_idx)
    torch.npu.synchronize()
    end = time.perf_counter()
    avg_time_us = (end - start) / 100 * 1000000.0
    if results:
        r = results[0]
        result_str = f"[{kernel_name}] B={B}, T={T}, D={D}, W={W}, avg_time={r['avg_us']:.2f}us, min_time={r['min_us']:.2f}us, max_time={r['max_us']:.2f}us"
    else:
        result_str = f'[{kernel_name}] B={B}, T={T}, D={D}, W={W}, avg_time=N/A (kernel not found in op_statistic.csv), perf_counter_avg={avg_time_us:.2f}us'
    print(result_str)
    with open(os.path.join(RESULT_DIR, 'benchmarks.result'), 'a') as f:
        f.write(result_str + '\n')
