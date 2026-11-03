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
from yb_b716f65a487720 import parallel_nsa_bwd_kernel_dq

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

def test_parallel_nsa_bwd_kernel_dq():
    """Test function for parallel_nsa_bwd_kernel_dq with NPU using npu/gpu/gpu_fp32 three-way comparison"""
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/parallel_nsa_bwd_kernel_dq_v2'
    inputs_dict = load_fixture(os.path.join(load_dir, 'inputs.pt'))
    device = torch.device('npu')
    B = inputs_dict['B_param']
    T = inputs_dict['T_param']
    H = inputs_dict['H_param']
    HQ = inputs_dict['HQ_param']
    G = inputs_dict['G_param']
    K = inputs_dict['K_param']
    V = inputs_dict['V_param']
    S = inputs_dict['S_param']
    BS = inputs_dict['BS_param']
    BK = inputs_dict['BK_param']
    BV = inputs_dict['BV_param']
    NV = inputs_dict['NV_param']
    block_counts = inputs_dict['block_counts']
    scale = inputs_dict['scale']
    cu_seqlens = inputs_dict['cu_seqlens']
    token_indices = inputs_dict['token_indices']
    print(f'Testing parallel_nsa_bwd_kernel_dq with config:')
    print(f'  B={B}, T={T}, H={H}, HQ={HQ}, G={G}')
    print(f'  K={K}, V={V}, S={S}')
    print(f'  BS={BS}, BK={BK}, BV={BV}, NV={NV}')
    print(f'  block_counts={block_counts}, scale={scale}')
    print(f'  Device: {device}')
    q = inputs_dict['q'].to(device)
    k = inputs_dict['k'].to(device)
    v = inputs_dict['v'].to(device)
    lse = inputs_dict['lse'].to(device)
    delta = inputs_dict['delta'].to(device)
    do = inputs_dict['do'].to(device)
    block_indices = inputs_dict['block_indices'].to(device)
    print('\nExecuting parallel_nsa_bwd_kernel_dq on NPU...')
    dq_npu = torch.empty(NV, *q.shape, dtype=q.dtype if NV == 1 else torch.float, device=device)
    grid = (T, NV, B * H)
    parallel_nsa_bwd_kernel_dq[grid](q=q, k=k, v=v, lse=lse, delta=delta, do=do, dq=dq_npu, block_indices=block_indices, block_counts=block_counts, cu_seqlens=cu_seqlens, token_indices=token_indices, scale=scale, T=T, B=B, H=H, HQ=HQ, G=G, K=K, V=V, S=S, BS=BS, BK=BK, BV=BV)
    dq_npu_result = dq_npu.sum(0)
    torch.npu.synchronize()
    print(f'NPU output: dq shape={dq_npu_result.shape}, dtype={dq_npu_result.dtype}')
    outputs_dict = load_fixture(os.path.join(load_dir, 'outputs.pt'))
    dq_gpu_fp32 = outputs_dict['dq_fp32'].to(device)
    dq_gpu = outputs_dict['dq'].to(device)
    print('\n' + '=' * 60)
    print('Comparing dq outputs (NPU vs GPU_fp32 vs GPU_fp16 three-way comparison)...')
    print('=' * 60)
    print('\n--- NPU vs GPU_fp32 ---')
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(dq_npu_result, dq_gpu_fp32)
    print('\n--- GPU_fp16 vs GPU_fp32 ---')
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(dq_gpu, dq_gpu_fp32)
    print('\n--- Accuracy ratio (NPU / GPU_fp16) ---')
    max_ratio = relative_max_npu / relative_max_gpu
    mean_ratio = relative_mean_npu / relative_mean_gpu
    rmse_ratio = rmse_npu / rmse_gpu
    print(f'dq 最大相对误差比: {max_ratio:.4f}')
    print(f'dq 平均相对误差比: {mean_ratio:.4f}')
    print(f'dq 均方根误差比: {rmse_ratio:.4f}')
    assert max_ratio <= 2, f'dq 最大相对误差比 check failed: {max_ratio:.4f} > 2'
    assert mean_ratio <= 1.2, f'dq 平均相对误差比 check failed: {mean_ratio:.4f} > 1.2'
    assert rmse_ratio <= 1.2, f'dq 均方根误差比 check failed: {rmse_ratio:.4f} > 1.2'
    print('\n' + '=' * 60)
    print('All output comparisons passed!')
    print('=' * 60)
    print(f'\nOutput shapes:')
    print(f'  dq: {dq_npu_result.shape}')
