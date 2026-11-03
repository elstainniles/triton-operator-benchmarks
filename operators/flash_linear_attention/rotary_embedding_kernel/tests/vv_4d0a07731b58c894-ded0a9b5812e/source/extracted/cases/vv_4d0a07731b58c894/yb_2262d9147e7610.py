
from opbench.fla_fixtures import load_fixture
import os
import csv
import glob
import time
import pytest
import torch
import torch_npu
from yb_df4e244f2240a0 import rotary_embedding_fwdbwd
from yb_df4e244f2240a0 import RotaryEmbedding
NPU_ID = os.environ.get('BENCH_NPU_ID', '0')
torch.npu.set_device(int(NPU_ID))
ACTIVE_ITERS = 20
WAIT_ITERS = 1
SCHEDULE_WARMUP = 1
SCHEDULE_SKIP_FIRST = 1
TOTAL_ITERS = SCHEDULE_SKIP_FIRST + WAIT_ITERS + SCHEDULE_WARMUP + ACTIVE_ITERS
experimental_config = torch_npu.profiler._ExperimentalConfig(aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization, profiler_level=torch_npu.profiler.ProfilerLevel.Level1, l2_cache=False)
B, T, H, D = (2, 4096, 4, 128)
kernel_name = 'rotary_embedding_kernel'

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
def test_rotary_embedding_kernel_perftest(load_dir):
    inputs = load_fixture(os.path.join(load_dir, f'{kernel_name}_in.pt'))
    x = inputs['x'].to('npu')
    cos = inputs['cos'].to('npu')
    sin = inputs['sin'].to('npu')
    result_dir = os.path.dirname(os.path.abspath(__file__))
    trace_dir = os.path.join(result_dir, f'prof_dir_{kernel_name}')
    with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.NPU], with_stack=False, record_shapes=False, profile_memory=False, schedule=torch_npu.profiler.schedule(wait=WAIT_ITERS, warmup=SCHEDULE_WARMUP, active=ACTIVE_ITERS, repeat=1, skip_first=SCHEDULE_SKIP_FIRST), experimental_config=experimental_config, on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(trace_dir)) as prof:
        for _ in range(TOTAL_ITERS):
            x_grad = x.requires_grad_(True)
            y = rotary_embedding_fwdbwd(x_grad, cos, sin)
            torch.npu.synchronize()
            prof.step()
        prof.stop()
    perf_iters = 50
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(perf_iters):
        x_grad = x.requires_grad_(True)
        y = rotary_embedding_fwdbwd(x_grad, cos, sin)
    torch.npu.synchronize()
    t1 = time.perf_counter()
    perf_avg_us = (t1 - t0) / perf_iters * 1000000.0
    csv_path = find_op_statistic_csv(trace_dir)
    if csv_path is None:
        print(f'WARNING: op_statistic.csv not found in {trace_dir}')
        return
    results = parse_op_statistic(csv_path, kernel_filter='rotary_embedding')
    if not results:
        all_results = parse_op_statistic(csv_path)
        print(f'No matching kernel found. All ops:')
        for r in all_results[:20]:
            print(f"  {r['op_type']}: avg={r['avg_us']:.2f}us")
        return
    r = results[0]
    shape_info = f'B={B}, T={T}, H={H}, D={D}'
    result_line = f"[{kernel_name}] shape={shape_info}, avg_time={r['avg_us']:.2f}us, min_time={r['min_us']:.2f}us, max_time={r['max_us']:.2f}us"
    print(result_line)
    print(f'perf_counter avg: {perf_avg_us:.2f}us')
