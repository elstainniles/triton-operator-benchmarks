# coding=utf-8
# -------------------------------------------------
# Project Name: TritonAscendTest
# File Name： test_chunk_bwd_kernel_dh_reduction_npu
# Tool Name: PyCharm
# Author : l00567229
# Date： 2026/5/19
# -------------------------------------------------
from opbench.fla_fixtures import load_fixture
import os
import pytest
import triton
import torch
import torch_npu
from yb_66eb8bea22f328 import chunk_bwd_kernel_dh_reduction
from yb_4524c847f154b7 import cal_precision

@pytest.mark.functiontest
def test_chunk_bwd_kernel_dh_reduction_npu(load_dir):
    data = load_fixture(os.path.join(load_dir, 'test_chunk_h_parallel_chunk_bwd_kernel_dh_reduction_v2.pt'), map_location=torch.device('cpu'), weights_only=False)
    data_fp64 = load_fixture(os.path.join(load_dir, 'test_chunk_h_parallel_chunk_bwd_kernel_dh_reduction_fp64_v2.pt'), map_location=torch.device('cpu'), weights_only=False)
    input_data = data['input_data']
    B = input_data['B']
    T = input_data['T']
    H = input_data['H']
    HQ = input_data['HQ']
    K = input_data['K']
    V = input_data['V']
    BT = input_data['BT']
    NG = input_data['NG']
    g_cumsum = input_data['g_cumsum'].to('npu')
    dh = input_data['dh'].to('npu')
    doq0 = input_data['doq0']
    h0 = input_data['h0']
    dh0 = torch.empty(B, HQ, K, V, dtype=torch.float32, device='npu') if h0 is not None else None

    def grid_reduction(meta):
        return (triton.cdiv(K, meta['BK']), triton.cdiv(V, meta['BV']), B * HQ)
    chunk_bwd_kernel_dh_reduction[grid_reduction](g=g_cumsum, gk=None, gv=None, dh=dh, doq0=doq0, dh0=dh0, cu_seqlens=None, chunk_offsets=None, T=T, HQ=HQ, H=H, K=K, V=V, BT=BT, NG=NG, USE_G=g_cumsum is not None, USE_GK=False, USE_GV=False)
    npu_output_dh = dh.cpu()
    gpu_output_dh_fp64 = data_fp64['gpu_output']['dh']
    gpu_output_dh_fp32 = data['gpu_output']['dh']
    print('\n=== NPU vs GPU fp64 ===')
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(npu_output_dh, gpu_output_dh_fp64)
    print('\n=== GPU fp32 vs GPU fp64 ===')
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(gpu_output_dh_fp32, gpu_output_dh_fp64)
    print(f'最大相对误差比 :{relative_max_npu / relative_max_gpu}, 平均相对误差比 :{relative_mean_npu / relative_mean_gpu}, 均方根误差比 :{rmse_npu / rmse_gpu}')
    assert relative_max_npu / relative_max_gpu <= 10, '最大相对误差比 check failed'
    assert relative_mean_npu / relative_mean_gpu <= 2, '平均相对误差比 check failed'
    assert rmse_npu / rmse_gpu <= 2, '均方根误差比 check failed'
