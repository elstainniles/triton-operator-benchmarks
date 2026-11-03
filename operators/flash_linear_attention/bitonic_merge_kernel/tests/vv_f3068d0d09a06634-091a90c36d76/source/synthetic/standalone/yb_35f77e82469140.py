# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from opbench.fla_fixtures import load_fixture
from typing import Optional
import torch
import triton
import triton.language as tl
import torch_npu

def validate_cmp(dtype, y_cal, y_ref, overflow_mode: Optional[str]=None, device_type: Optional[str]=None):
    if device_type is not None:
        target_device = torch.device(device_type)
        y_cal = y_cal.to(target_device)
        y_ref = y_ref.to(target_device)
    else:
        y_cal = y_cal.npu()
        y_ref = y_ref.npu()
    if overflow_mode == 'saturate':
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
        torch.testing.assert_close(y_ref, y_cal, rtol=0.001, atol=0.001, equal_nan=True)
    elif dtype == 'bfloat16':
        torch.testing.assert_close(y_ref.to(torch.float32), y_cal.to(torch.float32), rtol=0.005, atol=0.005, equal_nan=True)
    elif dtype == 'float32':
        torch.testing.assert_close(y_ref, y_cal, rtol=1e-05, atol=1e-05, equal_nan=True)
    elif dtype == 'int32' or dtype == 'int64' or dtype == 'int16' or (dtype == 'int8'):
        assert torch.equal(y_cal, y_ref)
    elif dtype == 'bool':
        assert torch.equal(y_cal, y_ref)
    else:
        raise ValueError('Invalid parameter "dtype" is found : {}'.format(dtype))

@triton.jit
def _compare_and_swap(x, ids, flip, i: tl.constexpr, n_dims: tl.constexpr):
    n_outer: tl.constexpr = x.numel >> n_dims
    shape: tl.constexpr = [n_outer * 2 ** i, 2, 2 ** (n_dims - i - 1)]
    y = tl.reshape(x, shape)
    mask = tl.arange(0, 2)[None, :, None]
    left = tl.broadcast_to(tl.sum(y * (1 - mask), 1)[:, None, :], shape).to(y.dtype)
    right = tl.broadcast_to(tl.sum(y * mask, 1)[:, None, :], shape).to(y.dtype)
    left = tl.reshape(left, x.shape)
    right = tl.reshape(right, x.shape)
    y_idx = tl.reshape(ids, shape)
    left_idx = tl.broadcast_to(tl.sum(y_idx * (1 - mask), 1)[:, None, :], shape)
    right_idx = tl.broadcast_to(tl.sum(y_idx * mask, 1)[:, None, :], shape)
    left_idx = tl.reshape(left_idx, x.shape).to(y_idx.dtype)
    right_idx = tl.reshape(right_idx, x.shape).to(y_idx.dtype)
    idtype = tl.core.get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)
    ileft = left.to(idtype, bitcast=True)
    iright = right.to(idtype, bitcast=True)
    ix = x.to(idtype, bitcast=True)
    cond = (left > right) != flip
    ret = ix ^ tl.where(cond, ileft ^ iright, tl.zeros_like(ix))
    new_ids = ids ^ tl.where(cond, left_idx ^ right_idx, tl.zeros_like(ids))
    return (ret.to(x.dtype, bitcast=True), new_ids)

@triton.jit
def _bitonic_merge(x, ids, stage: tl.constexpr, order: tl.constexpr, n_dims: tl.constexpr):
    n_outer: tl.constexpr = x.numel >> n_dims
    tl.static_assert(stage <= n_dims)
    if order == 2:
        shape: tl.constexpr = [n_outer * 2 ** (n_dims - 1 - stage), 2, 2 ** stage]
        flip = tl.reshape(tl.broadcast_to(tl.arange(0, 2)[None, :, None], shape), x.shape)
    else:
        flip = order
    for i in tl.static_range(stage):
        x, ids = _compare_and_swap(x, ids, flip, i + (n_dims - stage), n_dims)
    return (x, ids)

@triton.jit
def bitonic_merge_kernel(x_ptr, ids_ptr, output_x_ptr, output_ids_ptr, N: tl.constexpr, stage: tl.constexpr, order: tl.constexpr, n_dims: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """
    直接调用_bitonic_merge函数的wrapper kernel
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    ids = tl.load(ids_ptr + offsets, mask=mask)
    x_new, ids_new = _bitonic_merge(x, ids, stage, order, n_dims)
    tl.store(output_x_ptr + offsets, x_new, mask=mask)
    tl.store(output_ids_ptr + offsets, ids_new, mask=mask)

def test_bitonic_merge():
    """测试_bitonic_merge函数"""
    N = 4096
    block_size = 1024
    n_dims = 10
    stage = 1
    order = 0
    device = torch.device('npu:0')
    dtype = torch.float32
    input_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_vv_ops_2/_bitonic_merge'
    print(f'Loading inputs from: {input_dir}')
    x = load_fixture(f'{input_dir}/input_x.pt').to(device)
    ids = load_fixture(f'{input_dir}/input_ids.pt').to(device)
    print(f'测试配置: N={N}, n_dims={n_dims}, stage={stage}, order={order}')
    print(f'输入形状: x={x.shape}, ids={ids.shape}')
    output_x = torch.empty_like(x)
    output_ids = torch.empty_like(ids)
    bitonic_merge_kernel[triton.cdiv(N, block_size),](x, ids, output_x, output_ids, N, stage, order, n_dims, BLOCK_SIZE=block_size)
    torch.npu.synchronize()
    print('\nLoading expected outputs and comparing...')
    expected_output_x = load_fixture(f'{input_dir}/output_x.pt').to(device)
    expected_output_ids = load_fixture(f'{input_dir}/output_ids.pt').to(device)
    validate_cmp('float32', output_x, expected_output_x)
    validate_cmp('int32', output_ids, expected_output_ids)
    print('✓ All outputs matched')
    print(f'\n输出形状: output_x={output_x.shape}, output_ids={output_ids.shape}')
    print('_bitonic_merge测试通过！')
    return (output_x, output_ids)
