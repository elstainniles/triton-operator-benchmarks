import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import csv
import glob
import time
import pytest

import torch
import torch_npu

from fla.ops.utils.softmax import softmax_bwd
from fla.utils import IS_AMD, autotune_cache_kwargs

NPU_ID = os.environ.get("BENCH_NPU_ID", "0")
torch.npu.set_device(int(NPU_ID))
DEVICE = f"npu:{torch.npu.current_device()}"

WARMUP_ITERS = 5
ACTIVE_ITERS = 20
WAIT_ITERS = 1
SCHEDULE_WARMUP = 1
SCHEDULE_SKIP_FIRST = 1
TOTAL_ITERS = SCHEDULE_SKIP_FIRST + WAIT_ITERS + SCHEDULE_WARMUP + ACTIVE_ITERS

experimental_config = torch_npu.profiler._ExperimentalConfig(
    aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
    profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
    l2_cache=False
)

N, D = 1024, 256
kernel_name = "softmax_bwd_kernel"


def find_op_statistic_csv(prof_dir):
    pattern = os.path.join(prof_dir, "**", "ASCEND_PROFILER_OUTPUT", "op_statistic.csv")
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        return None
    matches.sort(key=os.path.getmtime, reverse=True)
    return matches[0]


def parse_op_statistic(csv_path, kernel_filter=None):
    results = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            op_type = row.get("OP Type", "")
            if kernel_filter and kernel_filter not in op_type:
                continue
            results.append({
                "op_type": op_type,
                "avg_us": float(row.get("Avg Time(us)", 0)),
                "max_us": float(row.get("Max Time(us)", 0)),
                "min_us": float(row.get("Min Time(us)", 0)),
                "count": int(row.get("Count", 0)),
            })
    return results


@pytest.mark.functiontest
def test_softmax_bwd_kernel_functiontest(load_dir):
    inputs = torch.load(os.path.join(load_dir, f"{kernel_name}_in.pt"))
    p = inputs['p'].to(DEVICE)
    dp = inputs['dp'].to(DEVICE)

    output = softmax_bwd(p, dp)

    expected_outputs = torch.load(os.path.join(load_dir, f"{kernel_name}_out.pt"))
    expected_output = expected_outputs['output'].to(DEVICE)

    dtype = p.dtype
    if dtype == torch.float32:
        atol, rtol = 1e-5, 1e-5
    elif dtype == torch.float16:
        atol, rtol = 1e-3, 1e-3
    elif dtype == torch.bfloat16:
        atol, rtol = 5e-3, 5e-3
    else:
        atol, rtol = 1e-5, 1e-5

    torch.testing.assert_close(output, expected_output, atol=atol, rtol=rtol)
    print("Output precision check passed")
    print(f"[{kernel_name}] Precision check completed successfully")


@pytest.mark.perftest
def test_softmax_bwd_kernel_perftest(load_dir):
    inputs = torch.load(os.path.join(load_dir, f"{kernel_name}_in.pt"))
    p = inputs['p'].to(DEVICE)
    dp = inputs['dp'].to(DEVICE)

    result_dir = os.path.dirname(os.path.abspath(__file__))
    trace_dir = os.path.join(result_dir, f"prof_dir_{kernel_name}")

    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        with_stack=False,
        record_shapes=False,
        profile_memory=False,
        schedule=torch_npu.profiler.schedule(
            wait=WAIT_ITERS,
            warmup=SCHEDULE_WARMUP,
            active=ACTIVE_ITERS,
            repeat=1,
            skip_first=SCHEDULE_SKIP_FIRST,
        ),
        experimental_config=experimental_config,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(trace_dir),
    ) as prof:
        for _ in range(TOTAL_ITERS):
            output = softmax_bwd(p, dp)
            torch.npu.synchronize()
            prof.step()
        prof.stop()

    perf_iters = 50
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(perf_iters):
        output = softmax_bwd(p, dp)
    torch.npu.synchronize()
    t1 = time.perf_counter()
    perf_avg_us = (t1 - t0) / perf_iters * 1e6

    csv_path = find_op_statistic_csv(trace_dir)
    if csv_path is None:
        print(f"WARNING: op_statistic.csv not found in {trace_dir}")
        return

    results = parse_op_statistic(csv_path, kernel_filter=kernel_name)
    if not results:
        all_results = parse_op_statistic(csv_path)
        print(f"No matching kernel found. All ops:")
        for r in all_results[:20]:
            print(f"  {r['op_type']}: avg={r['avg_us']:.2f}us")
        return

    r = results[0]
    shape_info = f"N={N}, D={D}"
    result_line = f"[{kernel_name}] shape={shape_info}, avg_time={r['avg_us']:.2f}us, min_time={r['min_us']:.2f}us, max_time={r['max_us']:.2f}us"
    print(result_line)
    print(f"perf_counter avg: {perf_avg_us:.2f}us")
