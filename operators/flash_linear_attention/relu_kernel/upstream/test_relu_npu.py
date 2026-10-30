import os
import sys
import csv
import glob
import pytest

import torch
import torch_npu
import triton
import triton.language as tl

from fla.utils import autotune_cache_kwargs

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

BATCH, M, N = 4, 1024, 2048
total_elements = BATCH * M * N

kernel_name = "relu"


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


def benchmark_kernel(kernel_name, func, args_func, result_dir):
    args = args_func()
    output = func(*args)
    torch.npu.synchronize()

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
            output = func(*args)
            torch.npu.synchronize()
            prof.step()
        prof.stop()

    csv_path = find_op_statistic_csv(trace_dir)
    if csv_path is None:
        print(f"WARNING: op_statistic.csv not found in {trace_dir}")
        return None
    return parse_op_statistic(csv_path, kernel_filter=kernel_name)


@triton.jit
def relu(x):
    return tl.maximum(x, 0.0)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': BS}, num_warps=num_warps)
        for BS in [256, 512, 1024, 2048]
        for num_warps in [1, 2, 4, 8]
    ],
    key=['N'],
    **autotune_cache_kwargs,
)
@triton.jit
def relu_kernel(
    input,
    output,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(input + offsets, mask=mask, other=0.0).to(tl.float32)
    y = relu(x)
    tl.store(output + offsets, y.to(output.dtype.element_ty), mask=mask)


def run_relu(input_tensor):
    N = input_tensor.numel()
    output = torch.empty_like(input_tensor)
    grid = (triton.cdiv(N, 1024),)
    relu_kernel[grid](input_tensor, output, N)
    return output


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': BS}, num_warps=num_warps)
        for BS in [256, 512, 1024, 2048]
        for num_warps in [1, 2, 4, 8]
    ],
    key=['N'],
    **autotune_cache_kwargs,
)
@triton.jit
def baseline_relu_kernel(
    input,
    output,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(input + offsets, mask=mask, other=0.0).to(tl.float32)
    y = x
    tl.store(output + offsets, y.to(output.dtype.element_ty), mask=mask)


def run_baseline_relu(input_tensor):
    N = input_tensor.numel()
    output = torch.empty_like(input_tensor)
    grid = (triton.cdiv(N, 1024),)
    baseline_relu_kernel[grid](input_tensor, output, N)
    return output


@pytest.mark.functiontest
def test_relu_kernel_functiontest(load_dir):
    inputs = torch.load(os.path.join(load_dir, f"{kernel_name}_in.pt"))
    x = inputs['x'].to(DEVICE)

    output = run_relu(x)

    expected_outputs = torch.load(os.path.join(load_dir, f"{kernel_name}_out.pt"))
    expected_output = expected_outputs['output'].to(DEVICE)
    dtype = x.dtype
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
def test_relu_kernel_perftest(load_dir):
    inputs = torch.load(os.path.join(load_dir, f"{kernel_name}_in.pt"))
    x = inputs['x'].to(DEVICE)

    result_dir = os.path.dirname(os.path.abspath(__file__))

    baseline_results = benchmark_kernel(
        kernel_name="baseline_relu_kernel",
        func=run_baseline_relu,
        args_func=lambda: (x,),
        result_dir=result_dir
    )

    relu_results = benchmark_kernel(
        kernel_name="relu_kernel",
        func=run_relu,
        args_func=lambda: (x,),
        result_dir=result_dir
    )

    shape_info = f"B={BATCH}, M={M}, N={N}, total_elements={total_elements}"

    if baseline_results and relu_results:
        baseline_avg = baseline_results[0]['avg_us']
        relu_avg = relu_results[0]['avg_us']
        compute_avg = relu_avg - baseline_avg
        print(f"[relu] shape={shape_info}, avg_time={compute_avg:.2f}us (total={relu_avg:.2f}us, baseline={baseline_avg:.2f}us)")
    else:
        print(f"[relu] shape={shape_info}, avg_time=N/A")
