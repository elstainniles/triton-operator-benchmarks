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
from yb_b13193c073f138 import parallel_path_bwd_dkv_fn

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

def test_parallel_path_bwd_dkv_fn():
    """Test function for parallel_path_bwd_dkv_fn with NPU"""
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/parallel_path_bwd_dkv_kernel'
    inputs_dict = load_fixture(os.path.join(load_dir, 'inputs.pt'))
    device = torch.device('npu')
    q_new_large = inputs_dict['q_new_large'].to(device)
    k_new_large = inputs_dict['k_new_large'].to(device)
    v = inputs_dict['v'].to(device)
    g_cumsum = inputs_dict['g_cumsum']
    if g_cumsum is not None:
        g_cumsum = g_cumsum.to(device)
    do = inputs_dict['do'].to(device)
    dv = inputs_dict['dv'].to(device)
    dg_cumsum = inputs_dict['dg_cumsum']
    if dg_cumsum is not None:
        dg_cumsum = dg_cumsum.to(device)
    hc_whole = inputs_dict['hc_whole'].to(device)
    scale = inputs_dict['scale']
    L = inputs_dict['L'].to(device)
    delta = inputs_dict['delta'].to(device)
    S = inputs_dict['S']
    BT = inputs_dict['BT']
    BS = inputs_dict['BS']
    B, T, num_blocks, HQ, K = q_new_large.shape
    H = k_new_large.shape[2]
    V = v.shape[-1]
    print(f'Testing parallel_path_bwd_dkv_fn with config:')
    print(f'  B={B}, T={T}, HQ={HQ}, H={H}, D={K}')
    print(f'  S={S}, BT={BT}, BS={BS}')
    print(f'  Device: {device}')
    print(f'  g_cumsum is None: {g_cumsum is None}')
    print('\nExecuting kernel on NPU...')
    dk_npu, dv_out_npu, dg_cumsum_out_npu = parallel_path_bwd_dkv_fn(q=q_new_large, k=k_new_large, v=v, g_cumsum=g_cumsum, do=do, dv=dv.clone(), dg_cumsum=dg_cumsum.clone() if dg_cumsum is not None else None, hc_whole=hc_whole, scale=scale, cu_seqlens=None, L=L, D=delta, S=S, BT=BT, BS=BS)
    torch.npu.synchronize()
    outputs_dict = load_fixture(os.path.join(load_dir, 'outputs.pt'))
    dk_gpu_fp32 = outputs_dict['dk_fp32'].to(device)
    dv_out_gpu_fp32 = outputs_dict['dv_out_fp32'].to(device)
    dk_gpu = outputs_dict['dk'].to(device)
    dv_out_gpu = outputs_dict['dv_out'].to(device)
    print('\n' + '=' * 60)
    print('Comparing dk outputs...')
    print('=' * 60)
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(dk_npu, dk_gpu_fp32)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(dk_gpu, dk_gpu_fp32)
    print(f'dk 最大相对误差比: {relative_max_npu / relative_max_gpu:.4f}')
    print(f'dk 平均相对误差比: {relative_mean_npu / relative_mean_gpu:.4f}')
    print(f'dk 均方根误差比: {rmse_npu / rmse_gpu:.4f}')
    assert relative_max_npu / relative_max_gpu <= 2, 'dk 最大相对误差比 check failed'
    assert relative_mean_npu / relative_mean_gpu <= 1.2, 'dk 平均相对误差比 check failed'
    assert rmse_npu / rmse_gpu <= 1.2, 'dk 均方根误差比 check failed'
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
    if dg_cumsum_out_npu is not None:
        dg_cumsum_out_gpu_fp32 = outputs_dict['dg_cumsum_out_fp32'].to(device)
        dg_cumsum_out_gpu = outputs_dict['dg_cumsum_out'].to(device)
        print('\n' + '=' * 60)
        print('Comparing dg_cumsum_out outputs...')
        print('=' * 60)
        relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(dg_cumsum_out_npu, dg_cumsum_out_gpu_fp32)
        relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(dg_cumsum_out_gpu, dg_cumsum_out_gpu_fp32)
        print(f'dg_cumsum_out 最大相对误差比: {relative_max_npu / relative_max_gpu:.4f}')
        print(f'dg_cumsum_out 平均相对误差比: {relative_mean_npu / relative_mean_gpu:.4f}')
        print(f'dg_cumsum_out 均方根误差比: {rmse_npu / rmse_gpu:.4f}')
        assert relative_max_npu / relative_max_gpu <= 2, 'dg_cumsum_out 最大相对误差比 check failed'
        assert relative_mean_npu / relative_mean_gpu <= 1.2, 'dg_cumsum_out 平均相对误差比 check failed'
        assert rmse_npu / rmse_gpu <= 1.2, 'dg_cumsum_out 均方根误差比 check failed'
    print('\n' + '=' * 60)
    print('All output comparisons passed!')
    print('=' * 60)
    print(f'\nOutput shapes:')
    print(f'  dk: {dk_npu.shape}')
    print(f'  dv_out: {dv_out_npu.shape}')
    if dg_cumsum_out_npu is not None:
        print(f'  dg_cumsum_out: {dg_cumsum_out_npu.shape}')
