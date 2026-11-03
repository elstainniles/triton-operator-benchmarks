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
from typing import Optional
from yb_e5a3ca015593ed import chunk_gla_fwd_A_kernel_intra_sub_inter
from yb_e5a3ca015593ed import chunk_gla_fwd_A_kernel_intra_sub_intra_split
from yb_e5a3ca015593ed import chunk_gla_fwd_A_kernel_intra_sub_intra_merge

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

def test_chunk_gla_fwd_A_kernel_intra_sub_intra_merge():
    """Test function for chunk_gla_fwd_A_kernel_intra_sub_intra_merge with NPU"""
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_vv_ops_2/chunk_gla_fwd_A_kernel_intra_sub_intra_merge'
    inputs_dict = load_fixture(os.path.join(load_dir, 'inputs.pt'))
    device = torch.device('npu')
    A_intra = inputs_dict['A_intra'].to(device)
    A = inputs_dict['A'].to(device)
    T = inputs_dict['T']
    B = inputs_dict['B']
    H = inputs_dict['H']
    BT = inputs_dict['BT']
    BC = inputs_dict['BC']
    BK = inputs_dict['BK']
    NK = inputs_dict['NK']
    NT = inputs_dict['NT']
    NC = inputs_dict['NC']
    print(f'Testing chunk_gla_fwd_A_kernel_intra_sub_intra_merge with config:')
    print(f'  B={B}, T={T}, H={H}')
    print(f'  BC={BC}, BK={BK}, NC={NC}, NT={NT}, NK={NK}')
    print(f'  Device: {device}')
    print('Step 3: Executing merge kernel on NPU...')
    grid_merge = (NT, NC, B * H)
    chunk_gla_fwd_A_kernel_intra_sub_intra_merge[grid_merge](A_intra, A, None, None, T=T, B=B, H=H, BT=BT, BC=BC, NK=NK)
    torch.npu.synchronize()
    A_npu = A.clone()
    outputs_dict = load_fixture(os.path.join(load_dir, 'outputs.pt'))
    outputs_dict_fp32 = load_fixture(os.path.join(load_dir, 'outputs_fp32.pt'))
    A_gpu_fp32 = outputs_dict_fp32['A_fp32'].to(device)
    A_gpu = outputs_dict['A'].to(device)
    print('\n' + '=' * 60)
    print('Comparing A outputs...')
    print('=' * 60)
    validate_cmp('float16', A_npu, A_gpu)
    print('\n' + '=' * 60)
    print('All output comparisons passed!')
    print('=' * 60)
    print(f'\nOutput shapes:')
    print(f'  A: {A_npu.shape}')
