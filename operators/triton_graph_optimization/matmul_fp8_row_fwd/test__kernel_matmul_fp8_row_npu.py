#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2012-2020. All rights reserved.
import sys
import os
from typing import Optional

import torch
import triton
import triton.language as tl
import torch_npu
import pytest

sys.path.append("..")

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
def _kernel_matmul_fp8_row(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
    m_key,
    n_key,
    k_key,
    A_scale,
    B_scale,
    Bias,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    dot_out_dtype: tl.constexpr,
    allow_tf32: tl.constexpr,
    fp8_fast_accum: tl.constexpr,
    skip_scaling_a: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    USE_BIAS: tl.constexpr,
    AB_DTYPE: tl.constexpr,
    NUM_SMS: tl.constexpr,
) -> None:
    """Matmul kernel of [M, K] @ [N, K] with row-wise scales

    performs swizzled matmul in [BLOCK_M, BLOCK_K] with [BLOCK_K, BLOCK_N] tiles.

    Args:
        A (TensorWrapper): [M, K] input tensor.
        B (TensorWrapper): [N, K] input tensor.
        C (TensorWrapper): [M, N] output tensor.
        M (int): M dimension of input tensor.
        N (int): N dimension of input tensor.
        K (int): K dimension of input tensor.
        m_key (int): Autotuning key for M dimension of input tensor.
        n_key (int): Autotuning key for N dimension of input tensor.
        k_key (int): Autotuning key for K dimension of input tensor.
        A_scale (TensorWrapper): [M] reciprocal scale tensor per row. A * A_scale = original A.
        B_scale (TensorWrapper): [N] reciprocal scale tensor per row. B * B_scale = original B.
        Bias (tensorWrapper): [N] Optional bias tensor.
        stride_am (int): Stride of M dimension of A.
        stride_ak (int): Stride of K dimension of A.
        stride_bn (int): Stride of N dimension of B.
        stride_bk (int): Stride of K dimension of B.
        stride_cm (int): Stride of M dimension of C.
        stride_cn (int): Stride of N dimension of C.
        dot_out_dtype (torch.dtype): Output type of tensor core.
        allow_tf32 (bool): Whether to use TF32 for tensor core.
        fp8_fast_accum (bool): Whether to use fast accumulation for tensor core.
        BLOCK_M (int): Block size for M dimension.
        BLOCK_N (int): Block size for N dimension.
        BLOCK_K (int): Block size for K dimension.
        GROUP_M (int): Number of groups for M dimension swizzle.
        SPLIT_K (int): Number of SM's to launch per row.
        USE_BIAS (bool): Whether to use bias.
        EVEN_K (bool): Whether K is evenly divisible by BLOCK_K * SPLIT_K.
        AB_DTYPE (bool): Whether to cast A and B to C.dtype before tensor core.
    """
    # Matrix multiplication.
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_tiles = num_pid_m * num_pid_n

    tiles_per_SM = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_SM += 1

    tile_id = start_pid - NUM_SMS
    ki = -1

    offs_k_for_mask = tl.arange(0, BLOCK_K)

    num_pid_in_group = GROUP_M * num_pid_n

    pid_m = 0
    pid_n = 0
    offs_am = tl.arange(0, BLOCK_M)
    offs_bn = tl.arange(0, BLOCK_N)
    acc_dtype = tl.float32 if allow_tf32 else dot_out_dtype
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)

    for _ in range(0, k_tiles * tiles_per_SM):
        ki = tl.where(ki == k_tiles - 1, 0, ki + 1)
        if ki == 0:
            tile_id += NUM_SMS
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

            start_m = pid_m * BLOCK_M
            start_n = pid_n * BLOCK_N
            offs_am = start_m + tl.arange(0, BLOCK_M)
            offs_bn = start_n + tl.arange(0, BLOCK_N)
            offs_am = tl.where(offs_am < M, offs_am, 0)
            offs_bn = tl.where(offs_bn < N, offs_bn, 0)
            offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_M), BLOCK_M)
            offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_N), BLOCK_N)
        offs_k = ki * BLOCK_K + tl.arange(0, BLOCK_K)
        A = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        B = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        a = tl.load(A, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_K, other=0.0)
        b = tl.load(B, mask=offs_k_for_mask[:, None] < K - ki * BLOCK_K, other=0.0)
        acc = tl.dot(a, b, acc, out_dtype=acc_dtype, allow_tf32=allow_tf32)

        if ki == k_tiles - 1:
            # rematerialize rm and rn to save registers
            rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

            # Invert scaling.
            b_scale = tl.load(B_scale + rn, mask=rn < N)
            if skip_scaling_a:
                acc *= b_scale[None, :]
            else:
                a_scale = tl.load(A_scale + rm, mask=rm < M)
                # pyre-ignore[16]: Undefined attribute [16]: `float`
                # has no attribute `__getitem__`.
                scale = a_scale[:, None] * b_scale[None, :]
                acc *= scale

            # Load and add bias if specified.
            if USE_BIAS:
                bias = tl.load(Bias + rn, mask=rn < N)
                acc += bias[None, :]

            acc = acc.to(C_ptr.dtype.element_ty)
            C = C_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
            mask = (rm < M)[:, None] & (rn < N)[None, :]
            # Handles write-back with reduction-splitting
            tl.store(C, acc, mask=mask)
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)

def convert_tensor_with_device_type(indata: dict, device_type: str):
    target_device = torch.device(device_type)
    outdata = {}

    for key, value in indata.items():
        if isinstance(value, torch.Tensor):
            if value.device.type != target_device.type:
                outdata[key] = value.to(target_device)
            else:
                outdata[key] = value
        else:
            outdata[key] = value

    return outdata

# 性能测试函数
def profiling_test(fn_triton, args=(), name="97", shape=(), tiling=()):
    experimental_config = torch_npu.profiler._ExperimentalConfig(
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1, l2_cache=False
        )
    with torch_npu.profiler.profile(
            activities=[  # torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU],
            with_stack=False,  # 采集torch 算子的函数调用栈的开关，该参数选填，默认关闭
            record_shapes=False,  # 采集torch 算子的input shape和input type的开关，该参数选填，默认关闭
            profile_memory=False,  # 采集memory相关数据的开关，该参数选填，默认关闭
            schedule=torch_npu.profiler.schedule(wait=1,
                                                warmup=1,
                                                active=30,
                                                repeat=1,
                                                skip_first=1),
            # warmup默认为0，老版本torch_npu包该参数为必填项
            experimental_config=experimental_config,  # 该参数选填，默认为Level0
            # 产生的profling文件的位置
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result_dir")
    ) as prof:
        # prof.start()
        for i in range(30):
            fn_triton(*args)
            torch.npu.synchronize()  # 确保 kernel 真正执行完
            prof.step()

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
    elif dtype == 'int32' or dtype == 'int64' or dtype == 'int16' or dtype == 'int8' or dtype == 'float8_e5m2':
        assert torch.equal(y_cal, y_ref)
    elif dtype == 'bool':
        assert torch.equal(y_cal, y_ref)
    else:
        raise ValueError('Invalid parameter \"dtype\" is found : {}'.format(dtype))

def cal_precision(input: torch.tensor, golden: torch.tensor):
    # 最大相对误差
    relative_max = torch.max(torch.abs(input-golden)/(torch.abs(golden)+float(1e-7)))
    print(f"max relative error: {relative_max}")
    # 平均相对误差
    relative_mean = torch.mean(torch.abs(input-golden)/(torch.abs(golden)+float(1e-7)))
    print(f"mean relative error: {relative_mean}")
    # 均方根误差
    rmse = torch.sqrt(torch.mean((input-golden)**2))
    print(f"rmse: {rmse}")
    # 小值域绝对误差
    mask = (torch.abs(golden) < float(2e-14) ) & (torch.abs(input-golden) > float(2e-30))
    error = torch.abs(input-golden)
    total_error = torch.sum(error[mask])
    print(f"small val error: {total_error}")

    return relative_max, relative_mean, rmse

def compare_data_precision(dict_ref: dict, dict_cal: dict, device_type: str):
    keys_ref, keys_cal = set(dict_ref.keys()), set(dict_cal.keys())
    if not keys_ref.issubset(keys_cal):
        raise ValueError("The keys of dict_ref is not subset of dict_cal")

    for key in dict_ref.keys():
        val_a, val_b = dict_ref[key], dict_cal[key]
        if type(val_a) != type(val_b):
            raise ValueError("The data type of two dicts are different")

        if isinstance(val_a, torch.Tensor):
            validate_cmp(dtype=str(val_a.dtype).split('.')[-1], y_ref=val_a, y_cal=val_b, device_type=device_type)
        else:
            raise ValueError("Non-tensor type is not currently supported")

def resolve_triton_dtype(input_data: dict) -> dict:
    """
    将 .pt 文件中保存的 dtype 字符串转换为 Triton JIT 编译器所需的真实 dtype 对象。
    必须在调用 kernel 之前执行！
    """
    dtype_val = input_data.get('TL_FP8_DTYPE')
    if isinstance(dtype_val, str):
        # 兼容带 "tl." 前缀和不带前缀的字符串
        attr_name = dtype_val.replace("tl.", "")
        if not hasattr(tl, attr_name):
            raise ValueError(f"Triton language does not have dtype: {attr_name}")
        input_data['TL_FP8_DTYPE'] = getattr(tl, attr_name)
    return input_data

def fn_triton(data, input_data):
    # 调用 kernel
    _kernel_matmul_fp8_row[data["grid"]](**input_data, force_simt_only=True, enable_graph_optimize=False)
    return


@pytest.mark.functiontest
def test_get_last_loc_kernel(ptfile_path):
    try:
        data = torch.load(ptfile_path, map_location=torch.device('cpu'), weights_only=False)
    except Exception as e:
        pytest.fail(f"load file {ptfile_path} failed: {str(e)}")

    input_data = convert_tensor_with_device_type(data["input_data"], device_type='npu')
    _kernel_matmul_fp8_row[data["grid"]](**input_data, force_simt_only=True, enable_graph_optimize=False)

    try:
        compare_data_precision(data["gpu_output"], input_data, device_type='cpu')
    except ValueError as e:
        pytest.fail(f"The testcase failed")


@pytest.mark.perftest
def test_perf(ptfile_path, ptfile_path1=None):
    try:
        data = torch.load(ptfile_path, map_location=torch.device('cpu'), weights_only=False)
    except Exception as e:
        pytest.fail(f"load file {ptfile_path} failed: {str(e)}")

    input_data = convert_tensor_with_device_type(data["input_data"], device_type='npu')
    profiling_test(fn_triton, (data, input_data))


if __name__ == '__main__':
    test_perf('test__kernel_matmul_fp8_row_v2_ieee.pt')