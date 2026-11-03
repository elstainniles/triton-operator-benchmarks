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
from yb_ca8319cd54ef79 import chunk_kda_bwd_kernel_wy_dqkg_fused

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

def test_chunk_kda_bwd_kernel_wy_dqkg_fused():
    """Test function for chunk_kda_bwd_kernel_wy_dqkg_fused with NPU"""
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/test_chunk_kda_bwd_kernel_wy_dqkg_fused'
    inputs_dict = load_fixture(os.path.join(load_dir, 'inputs.pt'))
    device = torch.device('npu')
    q = inputs_dict['q'].to(device)
    k = inputs_dict['k'].to(device)
    v = inputs_dict['v'].to(device)
    v_new = inputs_dict['v_new'].to(device)
    g = inputs_dict['g'].to(device)
    beta = inputs_dict['beta'].to(device)
    A = inputs_dict['A'].to(device)
    h = inputs_dict['h'].to(device)
    do = inputs_dict['do'].to(device)
    dh = inputs_dict['dh'].to(device)
    dv = inputs_dict['dv'].to(device)
    state_v_first = inputs_dict['state_v_first']
    scale = inputs_dict['scale']
    chunk_size = inputs_dict['chunk_size']
    B, T, H, K = k.shape
    HV, V = (v.shape[2], v.shape[-1])
    BT = chunk_size
    NT = triton.cdiv(T, BT)
    print(f'Testing chunk_kda_bwd_kernel_wy_dqkg_fused with config:')
    print(f'  B={B}, T={T}, H={H}, HV={HV}, K={K}, V={V}')
    print(f'  BT={BT}, NT={NT}')
    print(f'  Device: {device}')
    print(f'  state_v_first: {state_v_first}')
    dq_npu = g.new_empty(B, T, HV, K, dtype=torch.float)
    dk_npu = g.new_empty(B, T, HV, K, dtype=torch.float)
    dv2_npu = torch.empty_like(v)
    dg_npu = torch.empty_like(g, dtype=torch.float)
    db_npu = torch.empty_like(beta, dtype=torch.float)
    dA_npu = torch.empty_like(A, dtype=torch.float)
    print('\nExecuting kernel on NPU...')
    grid = (NT, B * HV)
    chunk_kda_bwd_kernel_wy_dqkg_fused[grid](q=q, k=k, v=v, v_new=v_new, g=g, beta=beta, A=A, h=h, do=do, dh=dh, dq=dq_npu, dk=dk_npu, dv=dv, dv2=dv2_npu, dg=dg_npu, db=db_npu, dA=dA_npu, cu_seqlens=None, chunk_indices=None, scale=scale, T=T, H=H, HV=HV, K=K, V=V, BT=BT, STATE_V_FIRST=state_v_first)
    torch.npu.synchronize()
    outputs_dict = load_fixture(os.path.join(load_dir, 'outputs.pt'))
    dq_gpu = outputs_dict['dq'].to(device)
    dk_gpu = outputs_dict['dk'].to(device)
    dv2_gpu = outputs_dict['dv2'].to(device)
    dg_gpu = outputs_dict['dg'].to(device)
    db_gpu = outputs_dict['db'].to(device)
    dA_gpu = outputs_dict['dA'].to(device)
    print('Comparing dq outputs...')
    validate_cmp('float16', dq_npu, dq_gpu)
    print('Comparing dk outputs...')
    validate_cmp('float16', dk_npu, dk_gpu)
    print('Comparing dv2 outputs...')
    validate_cmp('float16', dv2_npu, dv2_gpu)
    print('Comparing dg outputs...')
    validate_cmp('float16', dg_npu, dg_gpu)
    print('Comparing db outputs...')
    validate_cmp('float16', db_npu, db_gpu)
    print('Comparing dA outputs...')
    validate_cmp('float16', dA_npu, dA_gpu)
    print('\n' + '=' * 60)
    print('All output comparisons passed!')
    print('=' * 60)
    print(f'\nOutput shapes:')
    print(f'  dq: {dq_npu.shape}')
    print(f'  dk: {dk_npu.shape}')
    print(f'  dv2: {dv2_npu.shape}')
    print(f'  dg: {dg_npu.shape}')
    print(f'  db: {db_npu.shape}')
    print(f'  dA: {dA_npu.shape}')
