import os
import torch
import torch_npu
import triton
import triton.language as tl
from typing import Optional

RCP_LN2 = 1.4426950216


@triton.jit
def exp2(x):
    return tl.math.exp2(x.to(tl.float32))


@triton.heuristics({
    'USE_G': lambda args: args['g_cumsum'] is not None,
    'USE_WINDOW': lambda args: args['W'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def parallel_attn_bwd_kernel_dq(
        q,
        k,
        v,
        lse,
        delta,
        do,
        dq,
        dg_cumsum,
        g_cumsum,
        scale,
        cu_seqlens,
        chunk_indices,
        T,
        W: tl.constexpr,
        B: tl.constexpr,
        H: tl.constexpr,
        HQ: tl.constexpr,
        G: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BT: tl.constexpr,
        BS: tl.constexpr,
        BK: tl.constexpr,
        BV: tl.constexpr,
        USE_G: tl.constexpr,
        USE_WINDOW: tl.constexpr,
        IS_VARLEN: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        i_n = i_b
        bos, eos = i_n * T, i_n * T + T
    RCP_LN2: tl.constexpr = 1.4426950216

    p_q = tl.make_block_ptr(q + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_dq = tl.make_block_ptr(dq + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_do = tl.make_block_ptr(do + (bos * HQ + i_hq) * V, (T, V), (HQ * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_lse = tl.make_block_ptr(lse + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
    p_delta = tl.make_block_ptr(delta + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_do = tl.load(p_do, boundary_check=(0, 1))
    b_lse = tl.load(p_lse, boundary_check=(0,))
    b_delta = tl.load(p_delta, boundary_check=(0,))

    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    if USE_G:
        b_dg = tl.zeros([BT], dtype=tl.float32)
        p_gq = tl.make_block_ptr(g_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
        b_gq = tl.load(p_gq, boundary_check=(0,)).to(tl.float32)
    else:
        b_gq = None
        b_dg = None

    o_q = i_t * BT + tl.arange(0, BT)

    i_start = tl.maximum((i_t * BT - W + 1) // BS * BS, 0) if USE_WINDOW else 0

    for i_s in range(i_start, i_t * BT, BS):
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (K, T), (1, H * K), (0, i_s), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (V, T), (1, H * V), (i_v * BV, i_s), (BV, BS), (0, 1))

        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_s = tl.dot(b_q, b_k) * scale * RCP_LN2
        if USE_G:
            b_gk = tl.load(g_cumsum + (bos + o_k) * HQ + i_hq, mask=m_k, other=0).to(tl.float32)
            b_s += b_gq[:, None] - b_gk[None, :]

        if USE_WINDOW:
            b_s = tl.where((o_q[:, None] - o_k[None, :] < W) & m_k[None, :], b_s, float('-inf'))
        b_p = exp2(b_s - b_lse[:, None])
        b_dp = tl.dot(b_do, b_v)
        b_ds = b_p * (b_dp.to(tl.float32) - b_delta[:, None])
        b_dq += tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k))
        if USE_G:
            b_dg += tl.sum(b_ds, 1)

    for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (K, T), (1, H * K), (0, i_s), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (V, T), (1, H * V), (i_v * BV, i_s), (BV, BS), (0, 1))

        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_s = tl.dot(b_q, b_k) * scale * RCP_LN2

        if USE_G:
            p_gk = tl.make_block_ptr(g_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))
            b_gk = tl.load(p_gk, boundary_check=(0,)).to(tl.float32)
            b_s += b_gq[:, None] - b_gk[None, :]
        if USE_WINDOW:
            b_p = tl.where(
                (o_q[:, None] >= o_k[None, :]) & (o_q[:, None] - o_k[None, :] < W) & m_k[None, :],
                exp2(b_s - b_lse[:, None]), 0
            )
        else:
            b_p = tl.where((o_q[:, None] >= o_k[None, :]) & m_k[None, :], exp2(b_s - b_lse[:, None]), 0)

        b_dp = tl.dot(b_do, b_v)
        b_ds = b_p * (b_dp.to(tl.float32) - b_delta[:, None])
        b_dq += tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k))
        if USE_G:
            b_dg += tl.sum(b_ds, 1)

    b_dq *= scale
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    if USE_G:
        p_dg = tl.make_block_ptr(dg_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))


def cal_precision(input: torch.Tensor, golden: torch.Tensor):
    relative_max = torch.max(torch.abs(input - golden) / (torch.abs(golden) + float(1e-7)))
    print(f"max relative error: {relative_max}")
    relative_mean = torch.mean(torch.abs(input - golden) / (torch.abs(golden) + float(1e-7)))
    print(f"mean relative error: {relative_mean}")
    rmse = torch.sqrt(torch.mean((input - golden) ** 2))
    print(f"rmse: {rmse}")
    mask = (torch.abs(golden) < float(2e-14)) & (torch.abs(input - golden) > float(2e-30))
    error = torch.abs(input - golden)
    total_error = torch.sum(error[mask])
    print(f"small val error: {total_error}")
    return relative_max, relative_mean, rmse


def test_parallel_attn_bwd_kernel_dq():
    load_dir = "/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/parallel_attn_bwd_kernel_dq"
    inputs_dict = torch.load(os.path.join(load_dir, "inputs.pt"))

    device = torch.device('npu:0')
    q = inputs_dict['q'].to(device)
    k = inputs_dict['k'].to(device)
    v = inputs_dict['v'].to(device)
    lse = inputs_dict['lse'].to(device)
    delta = inputs_dict['delta'].to(device)
    do = inputs_dict['do'].to(device)
    dq = inputs_dict['dq'].to(device)
    dg_cumsum = inputs_dict['dg_cumsum']
    g_cumsum = inputs_dict['g_cumsum']
    scale = inputs_dict['scale']
    B = inputs_dict['B']
    T = inputs_dict['T']
    H = inputs_dict['H']
    HQ = inputs_dict['HQ']
    G = inputs_dict['G']
    K = inputs_dict['K']
    V = inputs_dict['V']
    BT = inputs_dict['BT']
    BS = inputs_dict['BS']
    BK = inputs_dict['BK']
    BV = inputs_dict['BV']

    NV = triton.cdiv(V, BV)
    NT = triton.cdiv(T, BT)
    grid = (NV, NT, B * HQ)

    print("\nExecuting kernel...")
    parallel_attn_bwd_kernel_dq[grid](
        q=q, k=k, v=v, lse=lse, delta=delta, do=do, dq=dq,
        dg_cumsum=dg_cumsum, g_cumsum=g_cumsum, scale=scale,
        cu_seqlens=None, chunk_indices=None, T=T, W=None,
        B=B, H=H, HQ=HQ, G=G, K=K, V=V, BT=BT, BS=BS, BK=BK, BV=BV,
        num_warps=4,
    )

    outputs_dict = torch.load(os.path.join(load_dir, "outputs.pt"))
    dq_gpu_fp64 = outputs_dict['dq_fp64'].to(device)
    dq_gpu = outputs_dict['dq'].to(device)

    print("\nComparing outputs with expected values...")
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(dq, dq_gpu_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(dq_gpu, dq_gpu_fp64)

    print(
        f"最大相对误差比 :{relative_max_npu / relative_max_gpu}, 平均相对误差比 :{relative_mean_npu / relative_mean_gpu}, 均方根误差比 :{rmse_npu / rmse_gpu}")
    assert relative_max_npu / relative_max_gpu <= 2, "最大相对误差比 check failed"
    assert relative_mean_npu / relative_mean_gpu <= 1.2, "平均相对误差比 check failed"
    assert rmse_npu / rmse_gpu <= 1.2, "均方根误差比 check failed"
    print("Output comparison passed!")


if __name__ == "__main__":
    test_parallel_attn_bwd_kernel_dq()
