
from opbench.fla_fixtures import load_fixture
import os
import csv
import glob
import pytest
import torch
import torch_npu
from yb_4f3456615af619 import prepare_position_ids_kernel
NPU_ID = os.environ.get('BENCH_NPU_ID', '0')
torch.npu.set_device(int(NPU_ID))
DEVICE = f'npu:{torch.npu.current_device()}'
ACTIVE_ITERS = 20
WAIT_ITERS = 1
SCHEDULE_WARMUP = 1
SCHEDULE_SKIP_FIRST = 1
TOTAL_ITERS = SCHEDULE_SKIP_FIRST + WAIT_ITERS + SCHEDULE_WARMUP + ACTIVE_ITERS
experimental_config = torch_npu.profiler._ExperimentalConfig(aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization, profiler_level=torch_npu.profiler.ProfilerLevel.Level1, l2_cache=False)
kernel_name = 'prepare_position_ids_kernel'

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

def benchmark_kernel(kernel_name, func, args_func, result_dir):
    args = args_func()
    output = func(*args)
    torch.npu.synchronize()
    trace_dir = os.path.join(result_dir, f'prof_dir_{kernel_name}')
    with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.NPU], with_stack=False, record_shapes=False, profile_memory=False, schedule=torch_npu.profiler.schedule(wait=WAIT_ITERS, warmup=SCHEDULE_WARMUP, active=ACTIVE_ITERS, repeat=1, skip_first=SCHEDULE_SKIP_FIRST), experimental_config=experimental_config, on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(trace_dir)) as prof:
        for _ in range(TOTAL_ITERS):
            output = func(*args)
            torch.npu.synchronize()
            prof.step()
        prof.stop()
    csv_path = find_op_statistic_csv(trace_dir)
    if csv_path is None:
        print(f'WARNING: op_statistic.csv not found in {trace_dir}')
        return None
    return parse_op_statistic(csv_path, kernel_filter=kernel_name)

def _get_value(val):
    return val.item() if isinstance(val, torch.Tensor) else val

@pytest.mark.perftest
def test_prepare_position_ids_kernel_perftest(load_dir):
    inputs = load_fixture(os.path.join(load_dir, f'{kernel_name}_in.pt'))
    y = inputs['y'].to(DEVICE)
    cu_seqlens = inputs['cu_seqlens'].to(DEVICE)
    B_KERNEL = _get_value(inputs['B_KERNEL'])
    B = _get_value(inputs['B'])

    def args_func():
        y_new = torch.empty(y.shape[0], device=DEVICE, dtype=torch.int32)
        return (y_new, cu_seqlens, B_KERNEL)

    def call_kernel(y, cu_seqlens, B):
        grid = (B,)
        prepare_position_ids_kernel[grid](y, cu_seqlens, B)
        return y
    result_dir = os.path.dirname(os.path.abspath(__file__))
    results = benchmark_kernel(kernel_name, call_kernel, args_func, result_dir)
    if results:
        for r in results:
            print(f"[{kernel_name}] B={B}, B_KERNEL={B_KERNEL}, avg_time={r['avg_us']:.2f}us, min_time={r['min_us']:.2f}us, max_time={r['max_us']:.2f}us")
    else:
        print('No results found')
