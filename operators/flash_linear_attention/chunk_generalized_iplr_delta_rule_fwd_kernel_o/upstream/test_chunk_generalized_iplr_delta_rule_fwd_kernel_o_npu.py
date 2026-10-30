import os
os.environ['FLA_USE_FAST_OPS'] = '0'

import torch
import torch_npu
import triton
import triton.language as tl
from typing import Optional

USE_CUDA_GRAPH = False
autotune_cache_kwargs = {}

BKV_LIST = [32, 64]


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK, 'BV': BV}, num_warps=num_warps)
        for BK in BKV_LIST
        for BV in BKV_LIST
        for num_warps in [2, 4]
    ],
    key=['BT'],
    use_cuda_graph=USE_CUDA_GRAPH,
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_generalized_iplr_delta_rule_fwd_kernel_o(
    q,
    k,
    v,
    u,
    b,
    h,
    o,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    b += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    u += (bos * H + i_h) * V
    o += (bos * H + i_h) * V
    h += (i_tg * H + i_h) * K * V
    stride_qk = H*K
    stride_vo = H*V

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_Aqk = tl.zeros([BT, BT], dtype=tl.float32)
    b_Aqb = tl.zeros([BT, BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k, (K, T), (1, stride_qk), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_h = tl.make_block_ptr(h, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        p_b = tl.make_block_ptr(b, (K, T), (1, stride_qk), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_b = tl.load(p_b, boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_o += tl.dot(b_q, b_h)
        b_Aqk += tl.dot(b_q, b_k)
        b_Aqb += tl.dot(b_q, b_b)

    o_i = tl.arange(0, BT)
    m_A = o_i[:, None] >= o_i[None, :]
    b_Aqk = tl.where(m_A, b_Aqk, 0)
    b_Aqb = tl.where(m_A, b_Aqb, 0)

    p_v = tl.make_block_ptr(v, (T, V), (stride_vo, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_u = tl.make_block_ptr(u, (T, V), (stride_vo, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o, (T, V), (stride_vo, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_u = tl.load(p_u, boundary_check=(0, 1))
    b_o = (b_o + tl.dot(b_Aqk.to(b_v.dtype), b_v) + tl.dot(b_Aqb.to(b_u.dtype), b_u)) * scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def chunk_generalized_iplr_delta_rule_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    v_new: torch.Tensor,
    b: torch.Tensor,
    h: torch.Tensor,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
) -> torch.Tensor:
    B, T, H, K, V = *q.shape, v.shape[-1]
    if scale is None:
        scale = K ** -0.5
    BT = chunk_size

    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices) if chunk_indices is not None else triton.cdiv(T, BT)

    o = torch.empty_like(v)

    def grid(meta): return (
        triton.cdiv(V, meta['BV']),
        NT,
        B * H,
    )
    chunk_generalized_iplr_delta_rule_fwd_kernel_o[grid](
        q=q,
        k=k,
        v=v,
        u=v_new,
        b=b,
        h=h,
        o=o,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
    )
    return o


def cal_precision(input: torch.Tensor, golden: torch.Tensor):
    relative_max = torch.max(torch.abs(input-golden)/(torch.abs(golden)+float(1e-7)))
    print(f"max relative error: {relative_max}")
    relative_mean = torch.mean(torch.abs(input-golden)/(torch.abs(golden)+float(1e-7)))
    print(f"mean relative error: {relative_mean}")
    rmse = torch.sqrt(torch.mean((input-golden)**2))
    print(f"rmse: {rmse}")
    mask = (torch.abs(golden) < float(2e-14) ) & (torch.abs(input-golden) > float(2e-30))
    error = torch.abs(input-golden)
    total_error = torch.sum(error[mask])
    print(f"small val error: {total_error}")
    return relative_max, relative_mean, rmse


def test_chunk_generalized_iplr_delta_rule_fwd_kernel_o():
    load_dir = "/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_generalized_iplr_delta_rule_fwd_kernel_o"
    inputs_dict = torch.load(os.path.join(load_dir, "inputs.pt"))
    
    device = torch.device('npu:0')
    q = inputs_dict['q'].to(device)
    k = inputs_dict['k'].to(device)
    v = inputs_dict['v'].to(device)
    v_new = inputs_dict['v_new'].to(device)
    b = inputs_dict['b'].to(device)
    h = inputs_dict['h'].to(device)
    scale = inputs_dict['scale']
    chunk_size = inputs_dict['chunk_size']
    
    print("\nExecuting kernel...")
    o_npu = chunk_generalized_iplr_delta_rule_fwd_o(
        q=q, k=k, v=v, v_new=v_new, b=b, h=h,
        scale=scale, cu_seqlens=None, chunk_size=chunk_size, chunk_indices=None
    )
    
    outputs_dict = torch.load(os.path.join(load_dir, "outputs.pt"))
    o_gpu_fp64 = outputs_dict['o_fp64'].to(device)
    o_gpu = outputs_dict['o'].to(device)
    
    print("\nComparing outputs with expected values...")
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(o_npu, o_gpu_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(o_gpu, o_gpu_fp64)
    
    print(f"最大相对误差比 :{relative_max_npu / relative_max_gpu}, 平均相对误差比 :{relative_mean_npu / relative_mean_gpu}, 均方根误差比 :{rmse_npu / rmse_gpu}")
    assert relative_max_npu / relative_max_gpu <= 2, "最大相对误差比 check failed"
    assert relative_mean_npu / relative_mean_gpu <= 1.2, "平均相对误差比 check failed"
    assert rmse_npu / rmse_gpu <= 1.2, "均方根误差比 check failed"
    print("Output comparison passed!")


if __name__ == "__main__":
    test_chunk_generalized_iplr_delta_rule_fwd_kernel_o()
