import os
import sys
import csv
import glob
import pytest

import torch
import torch_npu
import triton
import triton.language as tl
import triton.language.extra.cann.libdevice as tldevice

os.environ['FLA_USE_FAST_OPS'] = '0'

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

B, T, H, D = 2, 1024, 8, 1024
kernel_name = "tanh"


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
def tanh(x):
    return tldevice.tanh(x.to(tl.float32))


@triton.jit
def tanh_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tanh(x)
    tl.store(y_ptr + offsets, y, mask=mask)


@triton.jit
def baseline_tanh_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = x
    tl.store(y_ptr + offsets, y, mask=mask)


def tanh_call(x: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    tanh_kernel[grid](x, y, n_elements, BLOCK_SIZE=1024)
    return y


def baseline_tanh_call(x: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    baseline_tanh_kernel[grid](x, y, n_elements, BLOCK_SIZE=1024)
    return y


@pytest.mark.functiontest
def test_tanh_kernel_functiontest(load_dir):
    inputs = torch.load(os.path.join(load_dir, f"{kernel_name}_in.pt"))
    x = inputs['x'].to(DEVICE)

    output = tanh_call(x)

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
    print(f"[{kernel_name}] B={B}, T={T}, H={H}, D={D}, dtype={dtype}, precision check PASSED (atol={atol}, rtol={rtol})")


@pytest.mark.perftest
def test_tanh_kernel_perftest(load_dir):
    inputs = torch.load(os.path.join(load_dir, f"{kernel_name}_in.pt"))
    x = inputs['x'].to(DEVICE)

    result_dir = os.path.dirname(os.path.abspath(__file__))

    baseline_results = benchmark_kernel(
        kernel_name="baseline_tanh_kernel",
        func=baseline_tanh_call,
        args_func=lambda: (x,),
        result_dir=result_dir
    )

    tanh_results = benchmark_kernel(
        kernel_name="tanh_kernel",
        func=tanh_call,
        args_func=lambda: (x,),
        result_dir=result_dir
    )

    shape_info = f"B={B}, T={T}, H={H}, D={D}"

    if baseline_results and tanh_results:
        baseline_avg = baseline_results[0]['avg_us']
        tanh_avg = tanh_results[0]['avg_us']
        compute_avg = tanh_avg - baseline_avg
        print(f"[{kernel_name}] shape={shape_info}, avg_time={compute_avg:.2f}us (total={tanh_avg:.2f}us, baseline={baseline_avg:.2f}us)")
    else:
        print(f"[{kernel_name}] shape={shape_info}, avg_time=N/A")
