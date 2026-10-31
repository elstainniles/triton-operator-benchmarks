# coding=utf-8
# -------------------------------------------------
# Project Name: PythonProject1
# File Name： test_qk_rmsnorm_bias_triton_kernel_npu
# Tool Name: PyCharm
# Author : l00567229
# Date： 2025/12/20 14:19
# -------------------------------------------------
import sys
import pytest
import triton
import torch
import triton.language as tl
import torch_npu
import triton.language.extra.libdevice as tldevice
import os

# def get_npu_properties():
#     device = torch.npu.current_device()
#     print("using device: ", device)
#     return driver.active.utils.get_device_properties(device)
sys.path.append("..")
from utils import convert_tensor_with_device_type, validate_cmp, cal_precision, compare_data_precision


@triton.jit
def exp(x): 
    return tl.exp(x.to(tl.float32))


@triton.jit
def log(x): 
    return tl.log(x.to(tl.float32))


@triton.jit
def tanh(x): 
    return tldevice.tanh(x.to(tl.float32))


@triton.heuristics({
    "HAS_SMOOTHING": lambda args: args["label_smoothing"] > 0.0,
    "HAS_SOFTCAPPING": lambda args: args["logit_softcapping"] is not None,
})
@triton.jit
def cross_entropy_bwd_kernel(
    dlogits_ptr,
    dloss_ptr,
    logits_ptr,
    lse_ptr,
    labels_ptr,
    label_smoothing,
    logit_scale,
    lse_square_scale,
    logit_softcapping,
    ignore_index,
    total_classes,
    class_start_idx,
    n_cols,
    logits_row_stride,
    dlogits_row_stride,
    dloss_row_stride,
    BLOCK_SIZE: tl.constexpr,
    HAS_SMOOTHING: tl.constexpr,
    HAS_SOFTCAPPING: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)
    logits_ptr = logits_ptr + row_idx * logits_row_stride.to(tl.int64)
    dlogits_ptr = dlogits_ptr + row_idx * dlogits_row_stride.to(tl.int64)
    col_offsets = col_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    label_idx = tl.load(labels_ptr + row_idx)
    if label_idx != ignore_index:
        dloss = tl.load(dloss_ptr + row_idx * dloss_row_stride)
    else:
        dloss = 0.0
    logits = tl.load(logits_ptr + col_offsets, mask=col_offsets < n_cols, other=-float("inf")).to(
        tl.float32,
    ) * logit_scale
    if HAS_SOFTCAPPING:
        t = tanh(logits / logit_softcapping)
        logits = logit_softcapping * t
    lse = tl.load(lse_ptr + row_idx)
    probs = exp(logits - lse)
    probs += 2.0 * lse_square_scale * lse * probs
    label_idx -= class_start_idx
    if HAS_SMOOTHING:
        smooth_negative = label_smoothing / total_classes
        probs = tl.where(col_offsets == label_idx, probs - (1 - label_smoothing), probs) - smooth_negative
    else:
        probs = tl.where(col_offsets == label_idx, probs - 1.0, probs)
    if HAS_SOFTCAPPING:
        probs = probs * (1.0 - t * t)
    tl.store(dlogits_ptr + col_offsets, (dloss * logit_scale) * probs, mask=col_offsets < n_cols)


def fn_triton(data, input_data):
    # 调用 kernel
    cross_entropy_bwd_kernel[data["grid"]](**input_data)
    return

@pytest.mark.functiontest
def test_kernel():
    ptfile_path='/data/PytorchFile/Inference_or_Training_solution/fla_vv_ops_2/test_cross_entropy_bwd_kernel/test_cross_entropy_bwd_kernel.pt'
    try:
        data = torch.load(ptfile_path, map_location=torch.device('cpu'), weights_only=False)
    except Exception as e:
        pytest.fail(f"load file {ptfile_path} failed: {str(e)}")

    input_data = convert_tensor_with_device_type(data["input_data"], device_type='npu')
    cross_entropy_bwd_kernel[data["grid"]](**input_data)
    # profiling_test(fn_triton, (data, input_data))

    try:
        compare_data_precision(data["gpu_output"], input_data, device_type='cpu')
        print("pass")
    except ValueError as e:
        pytest.fail(f"The testcase failed")

@pytest.mark.perftest
def test_perf():
    ptfile_path='/data/PytorchFile/Inference_or_Training_solution/fla_vv_ops_2/test_cross_entropy_bwd_kernel/test_cross_entropy_bwd_kernel.pt'
    try:
        data = torch.load(ptfile_path, map_location=torch.device('cpu'), weights_only=False)
    except Exception as e:
        pytest.fail(f"load file {ptfile_path} failed: {str(e)}")

    input_data = convert_tensor_with_device_type(data["input_data"], device_type='npu')
    # profiling_test(fn_triton, (data, input_data))

if __name__ == "__main__":
    test_kernel('test_cross_entropy_bwd_kernel.pt')
