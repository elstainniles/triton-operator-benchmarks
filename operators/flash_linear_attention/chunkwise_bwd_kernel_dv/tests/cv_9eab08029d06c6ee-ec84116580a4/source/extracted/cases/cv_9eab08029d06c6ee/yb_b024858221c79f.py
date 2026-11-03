
from opbench.fla_fixtures import load_fixture
import os
import torch
import torch_npu
import triton
import triton.language as tl

@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({}, num_warps=num_warps, num_stages=num_stages) for num_warps in [4] for num_stages in [2, 3, 4]], key=['H', 'K', 'V'], restore_value=['dv'])
@triton.jit(do_not_specialize=['T'])
def chunkwise_bwd_kernel_dv(dh, k, g, dv, T, cu_seqlens, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, L: tl.constexpr, BT: tl.constexpr, NT: tl.constexpr, IS_VARLEN: tl.constexpr):
    i_t, i_nh = (tl.program_id(0), tl.program_id(1))
    i_n, i_h = (i_nh // H, i_nh % H)
    if IS_VARLEN:
        bos, eos = (tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32))
        T = eos - bos
    else:
        bos, eos = (i_n * T, i_n * T + T)
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    p_dh = tl.make_block_ptr(dh + ((i_n * NT + i_t) * H + i_h) * K * V, (K, V), (V, 1), (0, 0), (K, V), (1, 0))
    p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_k = tl.make_block_ptr(k + bos * K, (T, K), (K, 1), (i_t * BT, 0), (BT, K), (1, 0))
    p_dv = tl.make_block_ptr(dv + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_t * BT, 0), (BT, V), (1, 0))
    last_idx = min((i_t + 1) * BT, T) - 1
    b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
    b_dh = tl.load(p_dh, boundary_check=(0, 1))
    b_g = tl.load(p_g, boundary_check=(0,))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_dv = tl.where(m_t, tl.exp(-b_g + b_g_last), 0)[:, None] * tl.dot(b_k, b_dh).to(b_k.dtype)
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))

def cal_precision(input: torch.Tensor, golden: torch.Tensor):
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

def test_chunkwise_bwd_kernel_dv():
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunkwise_bwd_kernel_dv'
    inputs_dict = load_fixture(os.path.join(load_dir, 'inputs.pt'))
    device = torch.device('npu:0')
    dh = inputs_dict['dh'].to(device)
    k = inputs_dict['k'].to(device)
    g = inputs_dict['g'].to(device)
    dv = inputs_dict['dv'].to(device)
    B = inputs_dict['B']
    T = inputs_dict['T']
    H = inputs_dict['H']
    K = inputs_dict['K']
    V = inputs_dict['V']
    L = inputs_dict['L']
    BT = inputs_dict['BT']
    NT = inputs_dict['NT']
    grid = (NT, B * H)
    print('\nExecuting kernel...')
    chunkwise_bwd_kernel_dv[grid](dh=dh, k=k, g=g, dv=dv, cu_seqlens=None, T=T, H=H, K=K, V=V, L=L, BT=BT, NT=NT)
    outputs_dict = load_fixture(os.path.join(load_dir, 'outputs.pt'))
    dv_gpu_fp64 = outputs_dict['dv_fp64'].to(device)
    dv_gpu = outputs_dict['dv'].to(device)
    print('\nComparing outputs with expected values...')
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(dv, dv_gpu_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(dv_gpu, dv_gpu_fp64)
    print(f'最大相对误差比 :{relative_max_npu / relative_max_gpu}, 平均相对误差比 :{relative_mean_npu / relative_mean_gpu}, 均方根误差比 :{rmse_npu / rmse_gpu}')
    assert relative_max_npu / relative_max_gpu <= 2, '最大相对误差比 check failed'
    assert relative_mean_npu / relative_mean_gpu <= 1.2, '平均相对误差比 check failed'
    assert rmse_npu / rmse_gpu <= 1.2, '均方根误差比 check failed'
    print('Output comparison passed!')
