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
from yb_c00524fd58ccb9 import parallel_path_bwd_intra_chunk_fn
from yb_c00524fd58ccb9 import parallel_path_bwd_intra_chunk_kernel

def cal_precision(input: torch.tensor, golden: torch.tensor):
    relative_max = torch.max(torch.abs(input - golden) / (torch.abs(golden) + float(1e-07)))
    print(f'max relative error: {relative_max}')
    relative_mean = torch.mean(torch.abs(input - golden) / (torch.abs(golden) + float(1e-07)))
    print(f'mean relative error: {relative_mean}')
    rmse = torch.sqrt(torch.mean((input - golden) ** 2))
    print(f'rmse: {rmse}')
    mask = (torch.abs(golden) < float(2e-14)) & (torch.abs(input - golden) > float(2e-30))
    error = torch.abs(input - golden)
    total_error = torch.sum(error[mask])
    print(f'small val error: {total_error}')
    return (relative_max, relative_mean, rmse)

def test_parallel_path_bwd_intra_chunk_fn():
    """Test function for parallel_path_bwd_intra_chunk_fn with NPU"""
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/parallel_path_bwd_intra_chunk_kernel'
    inputs_dict = load_fixture(os.path.join(load_dir, 'inputs.pt'))
    device = torch.device('npu')
    q_new = inputs_dict['q_new'].to(device)
    k_new = inputs_dict['k_new'].to(device)
    v = inputs_dict['v'].to(device)
    g_cumsum = inputs_dict['g_cumsum']
    if g_cumsum is not None:
        g_cumsum = g_cumsum.to(device)
    w = inputs_dict['w'].to(device)
    h = inputs_dict['h'].to(device)
    L = inputs_dict['L'].to(device)
    delta = inputs_dict['delta'].to(device)
    scale = inputs_dict['scale']
    dw1 = inputs_dict['dw1'].to(device)
    dw2 = inputs_dict['dw2'].to(device)
    dq = inputs_dict['dq'].to(device)
    dk = inputs_dict['dk'].to(device)
    dv = inputs_dict['dv'].to(device)
    do = inputs_dict['do'].to(device)
    dg_cumsum = inputs_dict['dg_cumsum']
    if dg_cumsum is not None:
        dg_cumsum = dg_cumsum.to(device)
    S = inputs_dict['S']
    BT = inputs_dict['BT']
    B, T, HQ, D = q_new.shape
    H = k_new.shape[2]
    print(f'Testing parallel_path_bwd_intra_chunk_fn with config:')
    print(f'  B={B}, T={T}, HQ={HQ}, H={H}, D={D}')
    print(f'  S={S}, BT={BT}')
    print(f'  Device: {device}')
    print(f'  g_cumsum is None: {g_cumsum is None}')
    print('\nExecuting kernel on NPU...')
    dq_new_npu, dk_out_npu, dv_out_npu, dw1_out_npu, dw2_out_npu, dg_cumsum_out_npu = parallel_path_bwd_intra_chunk_fn(q=q_new, k=k_new, v=v, g_cumsum=g_cumsum, w1=w, w2=h, L=L, D=delta, scale=scale, dw1=dw1.clone(), dw2=dw2.clone(), dq=dq.clone(), dk=dk.clone(), dv=dv.clone(), do=do, dg_cumsum=dg_cumsum.clone() if dg_cumsum is not None else None, cu_seqlens=None, S=S, BT=BT)
    torch.npu.synchronize()
    outputs_dict = load_fixture(os.path.join(load_dir, 'outputs.pt'))
    dq_new_gpu_fp32 = outputs_dict['dq_new_fp32'].to(device)
    dk_out_gpu_fp32 = outputs_dict['dk_out_fp32'].to(device)
    dv_out_gpu_fp32 = outputs_dict['dv_out_fp32'].to(device)
    dw1_out_gpu_fp32 = outputs_dict['dw1_out_fp32'].to(device)
    dw2_out_gpu_fp32 = outputs_dict['dw2_out_fp32'].to(device)
    dq_new_gpu = outputs_dict['dq_new'].to(device)
    dk_out_gpu = outputs_dict['dk_out'].to(device)
    dv_out_gpu = outputs_dict['dv_out'].to(device)
    dw1_out_gpu = outputs_dict['dw1_out'].to(device)
    dw2_out_gpu = outputs_dict['dw2_out'].to(device)
    print('\n' + '=' * 60)
    print('Comparing dq_new outputs...')
    print('=' * 60)
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(dq_new_npu, dq_new_gpu_fp32)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(dq_new_gpu, dq_new_gpu_fp32)
    print(f'dq_new 最大相对误差比: {relative_max_npu / relative_max_gpu:.4f}')
    print(f'dq_new 平均相对误差比: {relative_mean_npu / relative_mean_gpu:.4f}')
    print(f'dq_new 均方根误差比: {rmse_npu / rmse_gpu:.4f}')
    assert relative_max_npu / relative_max_gpu <= 2, 'dq_new 最大相对误差比 check failed'
    assert relative_mean_npu / relative_mean_gpu <= 1.2, 'dq_new 平均相对误差比 check failed'
    assert rmse_npu / rmse_gpu <= 1.2, 'dq_new 均方根误差比 check failed'
    print('\n' + '=' * 60)
    print('Comparing dk_out outputs...')
    print('=' * 60)
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(dk_out_npu, dk_out_gpu_fp32)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(dk_out_gpu, dk_out_gpu_fp32)
    print(f'dk_out 最大相对误差比: {relative_max_npu / relative_max_gpu:.4f}')
    print(f'dk_out 平均相对误差比: {relative_mean_npu / relative_mean_gpu:.4f}')
    print(f'dk_out 均方根误差比: {rmse_npu / rmse_gpu:.4f}')
    assert relative_max_npu / relative_max_gpu <= 2, 'dk_out 最大相对误差比 check failed'
    assert relative_mean_npu / relative_mean_gpu <= 1.2, 'dk_out 平均相对误差比 check failed'
    assert rmse_npu / rmse_gpu <= 1.2, 'dk_out 均方根误差比 check failed'
    print('\n' + '=' * 60)
    print('Comparing dv_out outputs...')
    print('=' * 60)
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(dv_out_npu, dv_out_gpu_fp32)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(dv_out_gpu, dv_out_gpu_fp32)
    print(f'dv_out 最大相对误差比: {relative_max_npu / relative_max_gpu:.4f}')
    print(f'dv_out 平均相对误差比: {relative_mean_npu / relative_mean_gpu:.4f}')
    print(f'dv_out 均方根误差比: {rmse_npu / rmse_gpu:.4f}')
    assert relative_max_npu / relative_max_gpu <= 2, 'dv_out 最大相对误差比 check failed'
    assert relative_mean_npu / relative_mean_gpu <= 1.2, 'dv_out 平均相对误差比 check failed'
    assert rmse_npu / rmse_gpu <= 1.2, 'dv_out 均方根误差比 check failed'
    print('\n' + '=' * 60)
    print('Comparing dw1_out outputs...')
    print('=' * 60)
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(dw1_out_npu, dw1_out_gpu_fp32)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(dw1_out_gpu, dw1_out_gpu_fp32)
    print(f'dw1_out 最大相对误差比: {relative_max_npu / relative_max_gpu:.4f}')
    print(f'dw1_out 平均相对误差比: {relative_mean_npu / relative_mean_gpu:.4f}')
    print(f'dw1_out 均方根误差比: {rmse_npu / rmse_gpu:.4f}')
    assert relative_max_npu / relative_max_gpu <= 2, 'dw1_out 最大相对误差比 check failed'
    assert relative_mean_npu / relative_mean_gpu <= 1.2, 'dw1_out 平均相对误差比 check failed'
    assert rmse_npu / rmse_gpu <= 1.2, 'dw1_out 均方根误差比 check failed'
    print('\n' + '=' * 60)
    print('Comparing dw2_out outputs...')
    print('=' * 60)
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(dw2_out_npu, dw2_out_gpu_fp32)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(dw2_out_gpu, dw2_out_gpu_fp32)
    print(f'dw2_out 最大相对误差比: {relative_max_npu / relative_max_gpu:.4f}')
    print(f'dw2_out 平均相对误差比: {relative_mean_npu / relative_mean_gpu:.4f}')
    print(f'dw2_out 均方根误差比: {rmse_npu / rmse_gpu:.4f}')
    assert relative_max_npu / relative_max_gpu <= 2, 'dw2_out 最大相对误差比 check failed'
    assert relative_mean_npu / relative_mean_gpu <= 1.2, 'dw2_out 平均相对误差比 check failed'
    assert rmse_npu / rmse_gpu <= 1.2, 'dw2_out 均方根误差比 check failed'
    print('\n' + '=' * 60)
    print('All output comparisons passed!')
    print('=' * 60)
    print(f'\nOutput shapes:')
    print(f'  dq_new: {dq_new_npu.shape}')
    print(f'  dk_out: {dk_out_npu.shape}')
    print(f'  dv_out: {dv_out_npu.shape}')
    print(f'  dw1_out: {dw1_out_npu.shape}')
    print(f'  dw2_out: {dw2_out_npu.shape}')
