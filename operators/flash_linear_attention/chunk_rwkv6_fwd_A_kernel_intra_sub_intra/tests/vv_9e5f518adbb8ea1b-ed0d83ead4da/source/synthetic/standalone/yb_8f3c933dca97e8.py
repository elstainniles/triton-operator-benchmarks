# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from opbench.fla_fixtures import load_fixture
import os
import torch
import torch_npu
import triton
from typing import Optional
from yb_27b3e18bfc1d3a import chunk_rwkv6_fwd_A_kernel_intra_sub_inter
from yb_27b3e18bfc1d3a import chunk_rwkv6_fwd_A_kernel_intra_sub_intra

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

def test_chunk_rwkv6_fwd_A_kernel_intra_sub_intra():
    """Test function for chunk_rwkv6_fwd_A_kernel_intra_sub_intra with NPU"""
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_vv_ops_2/chunk_rwkv6_fwd_A_kernel_intra_sub_intra'
    inputs_dict = load_fixture(os.path.join(load_dir, 'inputs.pt'))
    device = torch.device('npu')
    q = inputs_dict['q'].to(device)
    k = inputs_dict['k'].to(device)
    gi = inputs_dict['gi'].to(device)
    ge = inputs_dict['ge'].to(device)
    u = inputs_dict['u'].to(device)
    A = inputs_dict['A'].to(device)
    scale = inputs_dict['scale']
    T = inputs_dict['T']
    H = inputs_dict['H']
    K = inputs_dict['K']
    BT = inputs_dict['BT']
    BC = inputs_dict['BC']
    BK = inputs_dict['BK']
    B, T, H, K = k.shape
    NC = triton.cdiv(BT, BC)
    NT = triton.cdiv(T, BT)
    print(f'Testing chunk_rwkv6_fwd_A_kernel_intra_sub_intra with config:')
    print(f'  B={B}, T={T}, H={H}, K={K}')
    print(f'  BT={BT}, BC={BC}, BK={BK}, NC={NC}, NT={NT}')
    print(f'  Device: {device}')
    print('\nExecuting intra-sub-intra kernel on NPU...')
    grid = (NT, NC, B * H)
    chunk_rwkv6_fwd_A_kernel_intra_sub_intra[grid](q=q, k=k, gi=gi, ge=ge, u=u, A=A, cu_seqlens=None, chunk_indices=None, scale=scale, T=T, H=H, K=K, BT=BT, BC=BC, BK=BK)
    torch.npu.synchronize()
    A_npu = A.clone()
    outputs_dict = load_fixture(os.path.join(load_dir, 'outputs.pt'))
    outputs_dict_fp32 = load_fixture(os.path.join(load_dir, 'outputs_fp32.pt'))
    A_gpu_fp32 = outputs_dict_fp32['A_out_fp32'].to(device)
    A_gpu = outputs_dict['A_out'].to(device)
    print('\n' + '=' * 60)
    print('Comparing A outputs...')
    print('=' * 60)
    validate_cmp('float16', A_npu, A_gpu)
    print('\n' + '=' * 60)
    print('All output comparisons passed!')
    print('=' * 60)
    print(f'\nOutput shapes:')
    print(f'  A: {A_npu.shape}')
