
from opbench.fla_fixtures import load_fixture
import os
import sys
import pytest
import csv
import glob
import time
import torch
import torch_npu
from yb_5a9aa3bd6a5163 import chunk_oja_fwd_h
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

@pytest.mark.perftest
def test_chunk_oja_fwd_kernel_h_blockdim64_performance():
    B, T, H, D = (4, 2048, 8, 128)
    K, V = (D, D)
    BT = 64
    dtype = torch.float32
    kernel_name = 'chunk_oja_fwd_kernel_h_blockdim64'
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_oja_fwd_kernel_h_blockdim64'
    print('Loading inputs from pt file...')
    inputs = load_fixture(os.path.join(load_dir, 'inputs.pt'))
    v = inputs['v'].to(DEVICE)
    w = inputs['w'].to(DEVICE)
    u = inputs['u'].to(DEVICE)
    gv = inputs['gv'].to(DEVICE)
    initial_state = inputs['initial_state'].to(DEVICE)
    output_final_state = inputs['output_final_state']
    chunk_size = inputs['chunk_size']
    result_dir = os.path.dirname(os.path.abspath(__file__))
    trace_dir = os.path.join(result_dir, f'prof_dir_chunk_oja_fwd_kernel_h_blockdim64')
    with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.NPU], with_stack=False, record_shapes=False, profile_memory=False, schedule=torch_npu.profiler.schedule(wait=WAIT_ITERS, warmup=SCHEDULE_WARMUP, active=ACTIVE_ITERS, repeat=1, skip_first=SCHEDULE_SKIP_FIRST), experimental_config=experimental_config, on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(trace_dir)) as prof:
        for _ in range(TOTAL_ITERS):
            h, k_new, final_state = chunk_oja_fwd_h(v, w, u, gv=gv, initial_state=initial_state, output_final_state=output_final_state, chunk_size=chunk_size)
            torch.npu.synchronize()
            prof.step()
        prof.stop()
    csv_path = find_op_statistic_csv(trace_dir)
    if csv_path is None:
        print(f'WARNING: op_statistic.csv not found in {trace_dir}')
        sys.exit(1)
    profiler_results = parse_op_statistic(csv_path, kernel_filter='chunk_oja_fwd_kernel_h_blockdim64')
    start_time = time.perf_counter()
    for _ in range(100):
        h, k_new, final_state = chunk_oja_fwd_h(v, w, u, gv=gv, initial_state=initial_state, output_final_state=output_final_state, chunk_size=chunk_size)
        torch.npu.synchronize()
    elapsed = (time.perf_counter() - start_time) / 100 * 1000000.0
    shape_info = f'B={B}, T={T}, H={H}, K={K}, V={V}, BT={BT}'
    if profiler_results:
        r = profiler_results[0]
        result_line = f"[chunk_oja_fwd_kernel_h_blockdim64] {shape_info}, avg_time={r['avg_us']:.2f}us, min_time={r['min_us']:.2f}us, max_time={r['max_us']:.2f}us"
    else:
        result_line = f'[chunk_oja_fwd_kernel_h_blockdim64] {shape_info}, avg_time={elapsed:.2f}us (perf_counter), profiler_result_not_found'
    print(result_line)
