import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import csv
import glob
import pytest

import torch
import torch_npu
import triton
import triton.language as tl

from fla.ops.utils.op import exp

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

B, T, H, D = 4, 2048, 8, 1024
kernel_name = "exp"


@triton.jit
def exp_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = exp(x)
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def baseline_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = x
    tl.store(output_ptr + offsets, output, mask=mask)


def exp_call(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    exp_kernel[grid](x, output, n_elements, BLOCK_SIZE=4096)
    return output


def baseline_call(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    baseline_kernel[grid](x, output, n_elements, BLOCK_SIZE=4096)
    return output


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


def benchmark_kernel(kernel_name_arg, func, args_func, result_dir):
    args = args_func()
    output = func(*args)
    torch.npu.synchronize()

    trace_dir = os.path.join(result_dir, f"prof_dir_{kernel_name_arg}")

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
            output = func(*args)
            torch.npu.synchronize()
            prof.step()
        prof.stop()

    csv_path = find_op_statistic_csv(trace_dir)
    if csv_path is None:
        print(f"WARNING: op_statistic.csv not found in {trace_dir}")
        return None
    return parse_op_statistic(csv_path, kernel_filter=kernel_name_arg)


@pytest.mark.functiontest
def test_exp_functiontest(load_dir):
    inputs = torch.load(os.path.join(load_dir, "inputs.pt"))
    x = inputs['x'].to(DEVICE)

    output = exp_call(x)
    baseline_output = baseline_call(x)
    torch.npu.synchronize()

    expected_outputs = torch.load(os.path.join(load_dir, "outputs.pt"))
    expected_output = expected_outputs['output'].to(DEVICE)
    expected_baseline_output = expected_outputs['baseline_output'].to(DEVICE)

    torch.testing.assert_close(output, expected_output, atol=0.008, rtol=0.008)
    print("output precision check passed")

    torch.testing.assert_close(baseline_output, expected_baseline_output, atol=0.008, rtol=0.008)
    print("baseline_output precision check passed")

    print(f"[{kernel_name}] Precision check completed successfully")


@pytest.mark.perftest
def test_exp_perftest(load_dir):
    inputs = torch.load(os.path.join(load_dir, "inputs.pt"))
    x = inputs['x'].to(DEVICE)

    result_dir = os.path.dirname(os.path.abspath(__file__))

    baseline_results = benchmark_kernel(
        kernel_name_arg="baseline_kernel",
        func=baseline_call,
        args_func=lambda: (x,),
        result_dir=result_dir
    )

    exp_results = benchmark_kernel(
        kernel_name_arg="exp_kernel",
        func=exp_call,
        args_func=lambda: (x,),
        result_dir=result_dir
    )

    if baseline_results and exp_results:
        baseline_avg = baseline_results[0]['avg_us']
        exp_avg = exp_results[0]['avg_us']
        compute_avg = exp_avg - baseline_avg
        print(f"[{kernel_name}] B={B}, T={T}, H={H}, D={D}, avg_time={compute_avg:.2f}us (total={exp_avg:.2f}us, baseline={baseline_avg:.2f}us)")
    else:
        print(f"[{kernel_name}] B={B}, T={T}, H={H}, D={D}, avg_time=N/A")
