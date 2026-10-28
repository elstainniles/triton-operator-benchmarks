# coding=utf-8
# -------------------------------------------------
# Project Name: PythonProject1
# File Name： test__fbgemm_gather_scale_fp8_rowwise_quant_dense_tokens_npu
# Tool Name: PyCharm
# Author : l00567229
# Date： 2026/1/22 20:20
# -------------------------------------------------
# generate.py
import sys
import pytest
import torch

import triton
import triton.language as tl
import torch_npu
from typing import Optional

sys.path.append("..")
from test_common import convert_tensor_with_device_type, profiling_test, validate_cmp, cal_precision, compare_data_precision


# Kernel Implementations
num_warps_range = [1, 2, 4, 8, 16, 32]
superblock_factor_range = [0, 1, 2, 4, 8, 16, 32]  # 2^0=1 到 2^5=32

# 生成满足约束条件的配置
configs = []
for n_w in num_warps_range:
    for sf in superblock_factor_range:
        # 约束条件: 1 <= 2^superblock_factor * num_warps <= 64
        product = sf * n_w
        if 1 <= product <= 64:
            configs.append(
                triton.Config(
                    {'superblock_factor': sf},
                    num_warps=n_w
                )
            )

# 应用自动调优
@triton.autotune(
    configs=configs,
    key=['BLOCK_D'],
)
@triton.jit
def _fbgemm_gather_scale_fp8_rowwise_quant_dense_tokens(
    output_ptr,
    output_scale_ptr,
    input_ptr,
    token_indices_ptr,
    expert_indices_ptr,
    scores_ptr,
    scale_ub_ptr,
    stride_t,
    stride_e,
    valid_token_count,
    D: tl.constexpr,
    TL_FP8_DTYPE: tl.constexpr,
    MAX_FP8: tl.constexpr,
    EPS: tl.constexpr,
    CLAMP_MAX: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    tl.static_assert(D % BLOCK_D == 0, "D must be a multiple of BLOCK_D")

    output_token_index = tl.program_id(0)

    if valid_token_count is not None:
        valid_token_count = tl.load(
            valid_token_count, None, eviction_policy="evict_last"
        )
        if output_token_index >= valid_token_count:
            return

    input_token_index = tl.load(
        token_indices_ptr + output_token_index, None, eviction_policy="evict_first"
    )
    input_expert_index = tl.load(
        expert_indices_ptr + output_token_index, None, eviction_policy="evict_first"
    )
    input_score = tl.load(
        scores_ptr + input_token_index * stride_t + input_expert_index * stride_e,
        None,
        eviction_policy="evict_first",
    ).to(tl.float32)

    row_max = 0.0
    in_2d_ptr = (
        input_ptr + input_token_index.to(tl.int64) * D + tl.arange(0, BLOCK_D)[:]
    )
    for _ in range(0, D, BLOCK_D):
        input_token_value = tl.load(
            in_2d_ptr,
            None,
            eviction_policy="evict_last",
        ).to(tl.float32)
        output_token_value = input_token_value * input_score

        tile_max = tl.max(tl.abs(output_token_value))
        row_max = tl.maximum(tile_max, row_max)
        in_2d_ptr += BLOCK_D

    # Clamp max value appropriately.
    if CLAMP_MAX:
        ub = tl.load(scale_ub_ptr, eviction_policy="evict_last")
        row_max = tl.clamp(row_max, EPS, ub)
    else:
        row_max = tl.maximum(row_max, EPS)

    # Scale and quantize.
    output_scale = MAX_FP8 / row_max
    tl.store(output_scale_ptr + output_token_index, 1.0 / output_scale)

    in_2d_ptr = (
        input_ptr + input_token_index.to(tl.int64) * D + tl.arange(0, BLOCK_D)[:]
    )
    out_2d_ptr = (
        output_ptr + output_token_index.to(tl.int64) * D + tl.arange(0, BLOCK_D)[:]
    )
    for _ in range(0, D, BLOCK_D):
        # Load from L2
        input_token_value = tl.load(
            in_2d_ptr,
            None,
            eviction_policy="evict_first",
        ).to(tl.float32)
        # Rematerilize
        output_token_value_fp8 = (input_token_value * input_score) * output_scale

        # Clamp A to fp8 range to make sure there's no overflow.
        # This is required for AMD. Nvidia's default saturation
        # handles it, but it's nice to have anyway.
        output_token_value_fp8 = tl.clamp(output_token_value_fp8, -MAX_FP8, MAX_FP8).to(
            TL_FP8_DTYPE
        )
        tl.store(
            out_2d_ptr,
            output_token_value_fp8,
            None,
            cache_modifier=".cg",
        )
        in_2d_ptr += BLOCK_D
        out_2d_ptr += BLOCK_D


def fn_triton(data, input_data):
    # 调用 kernel
    _fbgemm_gather_scale_fp8_rowwise_quant_dense_tokens[data["grid"]](**input_data, enable_graph_optimize=False)
    return

@pytest.mark.functiontest
def test_get_last_loc_kernel(ptfile_path):
    try:
        data = torch.load(ptfile_path, map_location=torch.device('cpu'), weights_only=False)
    except Exception as e:
        pytest.fail(f"load file {ptfile_path} failed: {str(e)}")

    input_data = convert_tensor_with_device_type(data["input_data"], device_type='npu')
    fn_triton(data, input_data)

    try:
        compare_data_precision(data["gpu_output"], input_data, device_type='cpu')
        print("pass")
    except ValueError as e:
        pytest.fail(f"The testcase failed")

@pytest.mark.perftest
def test_perf(ptfile_path):
    try:
        data = torch.load(ptfile_path, map_location=torch.device('cpu'), weights_only=False)
    except Exception as e:
        pytest.fail(f"load file {ptfile_path} failed: {str(e)}")

    input_data = convert_tensor_with_device_type(data["input_data"], device_type='npu')
    profiling_test(fn_triton, (data, input_data))


if __name__ == '__main__':
    test_get_last_loc_kernel('test__fbgemm_gather_scale_fp8_rowwise_quant_dense_tokens_v2.pt')