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
def _fbgemm_gather_scale_dense_tokens(
    out,
    x,
    token_indices,
    expert_indices,
    scores,
    stride_t,
    stride_e,
    valid_token_count,
    D: tl.constexpr,
    BLOCK_D_OUTER: tl.constexpr,
    BLOCK_D_INNER: tl.constexpr,
):
    output_token_index = tl.program_id(0)
    feature_offset = tl.program_id(1) * BLOCK_D_OUTER

    if valid_token_count is not None:
        valid_token_count = tl.load(
            valid_token_count, None, eviction_policy="evict_last"
        )
        if output_token_index >= valid_token_count:
            return

    input_token_index = tl.load(
        token_indices + output_token_index, None, eviction_policy="evict_last"
    )
    input_expert_index = tl.load(
        expert_indices + output_token_index, None, eviction_policy="evict_last"
    )

    input_score = tl.load(
        scores + input_token_index * stride_t + input_expert_index * stride_e,
        None,
        eviction_policy="evict_last",
    ).to(tl.float32)

    for _ in range(0, BLOCK_D_OUTER // BLOCK_D_INNER):
        input_token_value = tl.load(
            x
            + input_token_index.to(tl.int64) * D
            + feature_offset
            + tl.arange(0, BLOCK_D_INNER)[:],
            None,
        ).to(tl.float32)
        output_token_value = input_token_value * input_score

        tl.store(
            out
            + output_token_index.to(tl.int64) * D
            + feature_offset
            + tl.arange(0, BLOCK_D_INNER)[:],
            output_token_value,
            None,
        )
        feature_offset += BLOCK_D_INNER

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

def fn_triton(data, input_data):
    # 调用 kernel
    _fbgemm_gather_scale_dense_tokens[data["grid"]](**input_data, force_simt_only=True, enable_graph_optimize=False)
    #_fbgemm_gather_scale_dense_tokens[data["grid"]](**input_data)
    return


@pytest.mark.functiontest
def test_get_last_loc_kernel(ptfile_path):
    try:
        data = torch.load(ptfile_path, map_location=torch.device('cpu'), weights_only=False)
    except Exception as e:
        pytest.fail(f"load file {ptfile_path} failed: {str(e)}")

    input_data = convert_tensor_with_device_type(data["input_data"], device_type='npu')
    _fbgemm_gather_scale_dense_tokens[data["grid"]](**input_data, force_simt_only=True, enable_graph_optimize=False)
    fn_triton(data, input_data)

    try:
        compare_data_precision(data["gpu_output"], input_data, device_type='cpu')
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
    test_perf('test__fbgemm_gather_scale_dense_tokens_v2.pt')