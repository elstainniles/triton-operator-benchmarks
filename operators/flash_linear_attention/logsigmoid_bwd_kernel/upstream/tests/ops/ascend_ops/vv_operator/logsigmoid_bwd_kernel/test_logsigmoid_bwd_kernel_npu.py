import os
import sys
import csv
import glob
import time
import pytest

import torch
import torch_npu
from typing import Optional

from fla.modules.activations import logsigmoid_bwd

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

CONFIGS = [
    {"B": 4, "T": 2048, "D": 1024, "dtype": torch.float32, "temperature": 1.0, "desc": "temperature=1.0"},
    {"B": 4, "T": 2048, "D": 2048, "dtype": torch.bfloat16, "temperature": 0.5, "desc": "temperature=0.5"},
    {"B": 4, "T": 2048, "D": 2048, "dtype": torch.bfloat16, "temperature": 2.0, "desc": "temperature=2.0"},
]

kernel_name = "logsigmoid_bwd"


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


def validate_cmp(dtype, y_cal, y_ref, overflow_mode: Optional[str] = None, device_type: Optional[str] = None):
    if device_type is not None:
        target_device = torch.device(device_type)
        y_cal = y_cal.to(target_device)
        y_ref = y_ref.to(target_device)
    else:
        y_cal = y_cal.npu()
        y_ref = y_ref.npu()
    if overflow_mode == "saturate":
        if dtype in ['float32', 'float16']:
            min_value = -torch.finfo(dtype).min
            max_value = torch.finfo(dtype).max
        elif dtype in ['int32', 'int16', 'int8']:
            min_value = torch.iinfo(dtype).min
            max_value = torch.iinfo(dtype).max
        elif dtype == 'bool':
            min_value = 0
            max_value = 1
        else:
            raise ValueError('Invalid parameter "dtype" is found : {}'.format(dtype))
        y_ref = torch.clamp(y_ref, min=min_value, max=max_value)
    if dtype == 'float16':
        torch.testing.assert_close(y_ref, y_cal, rtol=1e-03, atol=1e-03, equal_nan=True)
    elif dtype == 'bfloat16':
        torch.testing.assert_close(y_ref.to(torch.float32), y_cal.to(torch.float32), rtol=5e-03, atol=5e-03,
                                   equal_nan=True)
    elif dtype == 'float32':
        torch.testing.assert_close(y_ref, y_cal, rtol=1e-05, atol=1e-05, equal_nan=True)
    elif dtype == 'int32' or dtype == 'int64' or dtype == 'int16' or dtype == 'int8':
        assert torch.equal(y_cal, y_ref)
    elif dtype == 'bool':
        assert torch.equal(y_cal, y_ref)
    else:
        raise ValueError('Invalid parameter \"dtype\" is found : {}'.format(dtype))


@pytest.mark.functiontest
def test_logsigmoid_bwd_kernel_functiontest(load_dir):
    for cfg in CONFIGS:
        B, T, D, dtype = cfg["B"], cfg["T"], cfg["D"], cfg["dtype"]
        temperature, desc = cfg["temperature"], cfg["desc"]

        inputs = torch.load(os.path.join(load_dir, f"{kernel_name}_B{B}_T{T}_D{D}_temp{temperature}_in.pt"))
        x = inputs['x'].to(DEVICE)
        dy = inputs['dy'].to(DEVICE)

        dx = logsigmoid_bwd(x, dy, temperature=temperature)

        expected_outputs = torch.load(os.path.join(load_dir, f"{kernel_name}_B{B}_T{T}_D{D}_temp{temperature}_out.pt"))
        expected_dx = expected_outputs['dx'].to(DEVICE)
        dtype_str = 'float32' if dtype == torch.float32 else 'bfloat16'
        validate_cmp(dtype_str, dx, expected_dx)
        print(f"Accuracy check passed for logsigmoid_bwd (B={B}, T={T}, D={D}, temp={temperature})")

    print(f"[{kernel_name}] Precision check completed successfully")


@pytest.mark.perftest
def test_logsigmoid_bwd_kernel_perftest(load_dir):
    for cfg in CONFIGS:
        B, T, D, dtype = cfg["B"], cfg["T"], cfg["D"], cfg["dtype"]
        temperature, desc = cfg["temperature"], cfg["desc"]

        inputs = torch.load(os.path.join(load_dir, f"{kernel_name}_B{B}_T{T}_D{D}_temp{temperature}_in.pt"))
        x = inputs['x'].to(DEVICE)
        dy = inputs['dy'].to(DEVICE)

        result_dir = os.path.dirname(os.path.abspath(__file__))
        trace_dir = os.path.join(result_dir, f"prof_dir_{kernel_name}_B{B}_T{T}_D{D}_temp{temperature}")

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
                dx = logsigmoid_bwd(x, dy, temperature=temperature)
                torch.npu.synchronize()
                prof.step()
            prof.stop()

        perf_iters = 50
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(perf_iters):
            dx = logsigmoid_bwd(x, dy, temperature=temperature)
        torch.npu.synchronize()
        t1 = time.perf_counter()
        perf_avg_us = (t1 - t0) / perf_iters * 1e6

        csv_path = find_op_statistic_csv(trace_dir)
        if csv_path is None:
            print(f"WARNING: op_statistic.csv not found in {trace_dir}")
            continue

        results = parse_op_statistic(csv_path, kernel_filter=kernel_name)
        if not results:
            all_results = parse_op_statistic(csv_path)
            print(f"No matching kernel found for {kernel_name}. All ops:")
            for r in all_results[:20]:
                print(f"  {r['op_type']}: avg={r['avg_us']:.2f}us")
            continue

        r = results[0]
        shape_info = f"B={B}, T={T}, D={D}, temperature={temperature}"
        result_line = f"[{kernel_name}] shape={shape_info}, avg_time={r['avg_us']:.2f}us, min_time={r['min_us']:.2f}us, max_time={r['max_us']:.2f}us"
        print(result_line)
        print(f"perf_counter avg: {perf_avg_us:.2f}us")
