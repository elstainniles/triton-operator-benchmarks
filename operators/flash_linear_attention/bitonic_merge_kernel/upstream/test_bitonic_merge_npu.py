# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from typing import Optional
import torch
import triton
import triton.language as tl
import torch_npu


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

# ============ Kernel定义 ============
@triton.jit
def _compare_and_swap(
    x,
    ids,
    flip,
    i: tl.constexpr,
    n_dims: tl.constexpr,
):
    n_outer: tl.constexpr = x.numel >> n_dims
    shape: tl.constexpr = [n_outer * 2**i, 2, 2**(n_dims - i - 1)]
    y = tl.reshape(x, shape)
    # slice left/right with 'stride' 2**(n_dims - i - 1)
    mask = tl.arange(0, 2)[None, :, None]
    left = tl.broadcast_to(tl.sum(y * (1 - mask), 1)[:, None, :], shape).to(y.dtype)
    right = tl.broadcast_to(tl.sum(y * mask, 1)[:, None, :], shape).to(y.dtype)
    left = tl.reshape(left, x.shape)
    right = tl.reshape(right, x.shape)
    # idx
    y_idx = tl.reshape(ids, shape)
    left_idx = tl.broadcast_to(tl.sum(y_idx * (1 - mask), 1)[:, None, :], shape)
    right_idx = tl.broadcast_to(tl.sum(y_idx * mask, 1)[:, None, :], shape)
    left_idx = tl.reshape(left_idx, x.shape).to(y_idx.dtype)
    right_idx = tl.reshape(right_idx, x.shape).to(y_idx.dtype)
    # actual compare-and-swap
    idtype = tl.core.get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)
    ileft = left.to(idtype, bitcast=True)
    iright = right.to(idtype, bitcast=True)
    ix = x.to(idtype, bitcast=True)

    cond = (left > right) != flip
    ret = ix ^ tl.where(cond, ileft ^ iright, tl.zeros_like(ix))
    new_ids = ids ^ tl.where(cond, left_idx ^ right_idx, tl.zeros_like(ids))
    return ret.to(x.dtype, bitcast=True), new_ids


@triton.jit
def _bitonic_merge(
    x,
    ids,
    stage: tl.constexpr,
    order: tl.constexpr,
    n_dims: tl.constexpr,
):
    n_outer: tl.constexpr = x.numel >> n_dims
    tl.static_assert(stage <= n_dims)
    # flip denotes whether to re-arrange sub-sequences of elements in ascending or
    # descending order.
    # if flip = 00000000... then all elements will be re-arranged ascendingly at this stage
    # if flip = 00110011... then all the elements will be re-arranged alternatingly (with
    # a stride of 2) at this stage
    if order == 2:
        shape: tl.constexpr = [n_outer * 2**(n_dims - 1 - stage), 2, 2**stage]
        flip = tl.reshape(tl.broadcast_to(tl.arange(0, 2)[None, :, None], shape), x.shape)
    else:
        flip = order
    # perform `stage` rounds of `compare-and-swap`
    for i in tl.static_range(stage):
        x, ids = _compare_and_swap(x, ids, flip, i + (n_dims - stage), n_dims)
    return x, ids


# ============ Wrapper Kernel（直接调用_bitonic_merge）============
@triton.jit
def bitonic_merge_kernel(
    x_ptr,
    ids_ptr,
    output_x_ptr,
    output_ids_ptr,
    N: tl.constexpr,
    stage: tl.constexpr,
    order: tl.constexpr,
    n_dims: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    直接调用_bitonic_merge函数的wrapper kernel
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    # 加载输入
    x = tl.load(x_ptr + offsets, mask=mask)
    ids = tl.load(ids_ptr + offsets, mask=mask)

    # 直接调用_bitonic_merge
    x_new, ids_new = _bitonic_merge(x, ids, stage, order, n_dims)
    
    # 存储输出
    tl.store(output_x_ptr + offsets, x_new, mask=mask)
    tl.store(output_ids_ptr + offsets, ids_new, mask=mask)


# ============ 测试用例 ============
def test_bitonic_merge():
    """测试_bitonic_merge函数"""
    # 配置参数 - 使用较大的shape以便测试性能
    N = 4096  # 元素数量（必须是2的幂次）
    block_size = 1024
    n_dims = 10  # log2(4096) = 12
    stage = 1  # 第一阶段
    order = 0  # 升序
    
    device = torch.device('npu:0')
    dtype = torch.float32
    
    # Load input tensors from pt files
    input_dir = "/data/PytorchFile/Inference_or_Training_solution/fla_vv_ops_2/_bitonic_merge"
    print(f"Loading inputs from: {input_dir}")
    
    x = torch.load(f"{input_dir}/input_x.pt").to(device)
    ids = torch.load(f"{input_dir}/input_ids.pt").to(device)
    
    print(f"测试配置: N={N}, n_dims={n_dims}, stage={stage}, order={order}")
    print(f"输入形状: x={x.shape}, ids={ids.shape}")

    # 准备输出
    output_x = torch.empty_like(x)
    output_ids = torch.empty_like(ids)
    
    # 直接调用wrapper kernel
    bitonic_merge_kernel[(triton.cdiv(N, block_size),)](
        x, ids, output_x, output_ids, N, stage, order, n_dims, BLOCK_SIZE=block_size
    )
    torch.npu.synchronize()
    
    # Load expected outputs and compare
    print("\nLoading expected outputs and comparing...")
    expected_output_x = torch.load(f"{input_dir}/output_x.pt").to(device)
    expected_output_ids = torch.load(f"{input_dir}/output_ids.pt").to(device)
    
    # Compare outputs with tolerance
    validate_cmp("float32", output_x, expected_output_x)
    validate_cmp("int32", output_ids, expected_output_ids)
    
    print("✓ All outputs matched")
    
    print(f"\n输出形状: output_x={output_x.shape}, output_ids={output_ids.shape}")
    print("_bitonic_merge测试通过！")
    
    return output_x, output_ids


# ============ Profiling测试 ============
def profile_bitonic_merge():
    """对_bitonic_merge进行profiling"""
    # 配置参数 - 使用较大的shape以便测试性能
    N = 4096  # 元素数量
    block_size = 1024
    n_dims = 10  # log2(4096) = 12
    stage = 1  # 第一阶段
    order = 0  # 升序
    
    device = torch.device('npu:0')
    dtype = torch.float32
    
    # Load input tensors from pt files
    input_dir = "/data/PytorchFile/Inference_or_Training_solution/fla_vv_ops_2/_bitonic_merge"
    
    x = torch.load(f"{input_dir}/input_x.pt").to(device)
    ids = torch.load(f"{input_dir}/input_ids.pt").to(device)
    
    print(f"\n开始Profiling测试...")
    print(f"测试配置: N={N}, n_dims={n_dims}, stage={stage}, order={order}")
    
    output_x = torch.empty_like(x)
    output_ids = torch.empty_like(ids)
    
    # 配置profiler
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1, 
        l2_cache=False
    )
    
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        with_stack=False,
        record_shapes=False,
        profile_memory=False,
        schedule=torch_npu.profiler.schedule(
            wait=1,
            warmup=1,
            active=30,
            repeat=1,
            skip_first=1
        ),
        experimental_config=experimental_config,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result_dir_original")
    ) as prof:
        prof.start()
        for iter_idx in range(35):
            # 直接调用wrapper kernel
            bitonic_merge_kernel[(triton.cdiv(N, block_size),)](
                x, ids, output_x, output_ids, N, stage, order, n_dims, BLOCK_SIZE=block_size
            )
            torch.npu.synchronize()  # 确保 kernel 真正执行完
            prof.step()
        prof.stop()
    
    print("Profiling完成！结果已保存到 ./result_dir_original")


if __name__ == "__main__":
    # 先运行基本测试
    print("=" * 60)
    print("运行基本功能测试")
    print("=" * 60)
    test_bitonic_merge()
    
    # 再运行profiling测试
    print("\n" + "=" * 60)
    print("运行Profiling测试")
    print("=" * 60)
    profile_bitonic_merge()
