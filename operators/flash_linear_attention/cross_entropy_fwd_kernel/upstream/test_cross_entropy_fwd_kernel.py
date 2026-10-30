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
def cross_entropy_fwd_kernel(
    loss_ptr,
    lse_ptr,
    z_loss_ptr,
    logits_ptr,
    labels_ptr,
    label_smoothing,
    logit_scale,
    lse_square_scale,
    logit_softcapping,
    ignore_index,
    total_classes,
    class_start_idx,
    n_cols,
    n_rows,
    logits_row_stride,
    BLOCK_SIZE: tl.constexpr,
    HAS_SMOOTHING: tl.constexpr,
    HAS_SOFTCAPPING: tl.constexpr,
    SPLIT: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)
    logits_ptr = logits_ptr + row_idx * logits_row_stride.to(tl.int64)
    col_offsets = col_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    label_idx = tl.load(labels_ptr + row_idx)
    logits = tl.load(logits_ptr + col_offsets, mask=col_offsets < n_cols, other=-float("inf"))
    logits = logits.to(tl.float32) * logit_scale
    if HAS_SOFTCAPPING:
        logits = logit_softcapping * tanh(logits / logit_softcapping)
    max_logits = tl.max(logits, 0)
    if HAS_SMOOTHING:
        sum_logits = tl.sum(tl.where(col_offsets < n_cols, logits, 0.0), 0)
    lse = log(tl.sum(exp(logits - max_logits), 0)) + max_logits
    tl.store(lse_ptr + col_block_idx * n_rows + row_idx, lse)
    if label_idx == ignore_index:
        loss = 0.0
        z_loss = 0.0
    else:
        label_idx -= class_start_idx
        if label_idx >= col_block_idx * BLOCK_SIZE and label_idx < min(
            n_cols, (col_block_idx + 1) * BLOCK_SIZE,
        ):
            logits_label = tl.load(logits_ptr + label_idx).to(tl.float32) * logit_scale
            if HAS_SOFTCAPPING:
                logits_label = logit_softcapping * tanh(logits_label / logit_softcapping)
            if HAS_SMOOTHING:
                loss = (
                    (lse if not SPLIT else 0.0)
                    - label_smoothing * sum_logits / total_classes
                    - (1 - label_smoothing) * logits_label
                )
            else:
                loss = (lse if not SPLIT else 0.0) - logits_label
        else:
            if HAS_SMOOTHING:
                loss = label_smoothing * ((lse if not SPLIT else 0.0) - sum_logits / total_classes)
            else:
                loss = 0.0
        if not SPLIT:
            z_loss = lse_square_scale * lse * lse
            loss += z_loss
        else:
            z_loss = 0.0
    tl.store(loss_ptr + col_block_idx * n_rows + row_idx, loss)
    if not SPLIT:
        tl.store(z_loss_ptr + col_block_idx * n_rows + row_idx, z_loss)


def fn_triton(data, input_data):
    # 调用 kernel
    cross_entropy_fwd_kernel[data["grid"]](**input_data)
    return

@pytest.mark.functiontest
def test_kernel():
    ptfile_path='/data/PytorchFile/Inference_or_Training_solution/fla_vv_ops_2/test_cross_entropy_fwd_kernel/test_cross_entropy_fwd_kernel.pt'
    try:
        data = torch.load(ptfile_path, map_location=torch.device('cpu'), weights_only=False)
    except Exception as e:
        pytest.fail(f"load file {ptfile_path} failed: {str(e)}")

    input_data = convert_tensor_with_device_type(data["input_data"], device_type='npu')
    cross_entropy_fwd_kernel[data["grid"]](**input_data)
    # profiling_test(fn_triton, (data, input_data))

    try:
        compare_data_precision(data["gpu_output"], input_data, device_type='cpu')
        print("pass")
    except ValueError as e:
        pytest.fail(f"The testcase failed")

@pytest.mark.perftest
def test_perf():
    ptfile_path='/data/PytorchFile/Inference_or_Training_solution/fla_vv_ops_2/test_cross_entropy_fwd_kernel/test_cross_entropy_fwd_kernel.pt'
    try:
        data = torch.load(ptfile_path, map_location=torch.device('cpu'), weights_only=False)
    except Exception as e:
        pytest.fail(f"load file {ptfile_path} failed: {str(e)}")

    input_data = convert_tensor_with_device_type(data["input_data"], device_type='npu')
    # profiling_test(fn_triton, (data, input_data))

if __name__ == "__main__":
    test_kernel('test_cross_entropy_fwd_kernel.pt')
