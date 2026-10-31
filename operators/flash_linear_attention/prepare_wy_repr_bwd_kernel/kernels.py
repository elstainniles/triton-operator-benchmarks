# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: fla/ops/comba/wy_fast.py:159 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
# Source archive: dependencies and launch adaptation are not executable here.
# The complete original file and dependency snapshots are under upstream/.
import triton
import triton.language as tl


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'K', 'V', 'BT', 'BK', 'BV', 'IS_VARLEN'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def prepare_wy_repr_bwd_kernel(
    k,
    v,
    p,
    beta,
    g0,
    g,
    A,
    dw,
    du,
    dk,
    dv,
    dp,
    dbeta,
    dg0,
    dg,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    p_beta = tl.make_block_ptr(beta + (bos*H + i_h), (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_g0 = tl.make_block_ptr(g0 + (bos*H + i_h), (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_g = tl.make_block_ptr(g + (bos*H + i_h), (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_A = tl.make_block_ptr(A + (bos*H + i_h) * BT, (BT, T), (1, H*BT), (0, i_t * BT), (BT, BT), (0, 1))

    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_beta = tl.load(p_beta, boundary_check=(0,))
    b_g0 = tl.load(p_g0, boundary_check=(0,))
    b_g0_exp = exp2(b_g0)
    b_g = tl.load(p_g, boundary_check=(0,))

    b_dbeta = tl.zeros([BT], dtype=tl.float32)
    b_dA = tl.zeros([BT, BT], dtype=tl.float32)
    b_dg0 = tl.zeros([BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        p_p = tl.make_block_ptr(p + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dp = tl.make_block_ptr(dp + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dw = tl.make_block_ptr(dw + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_p = tl.load(p_p, boundary_check=(0, 1))
        b_p_beta_g0 = (b_p * b_beta[:, None] * b_g0_exp[:, None]).to(b_p.dtype)
        b_dw = tl.load(p_dw, boundary_check=(0, 1))
        b_dA += tl.dot(b_dw, tl.trans(b_p_beta_g0))
        b_dp_beta_g0 = tl.dot(b_A, b_dw)
        b_dp = b_dp_beta_g0 * b_beta[:, None] * b_g0_exp[:, None]
        b_dbeta += tl.sum(b_dp_beta_g0 * b_p * b_g0_exp[:, None], 1)
        b_dg0 += tl.sum(b_dp * b_p, 1)
        tl.store(p_dp, b_dp.to(p_dp.dtype.element_ty), boundary_check=(0, 1))

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_du = tl.make_block_ptr(du + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_v_beta = (b_v * b_beta[:, None]).to(b_v.dtype)
        b_du = tl.load(p_du, boundary_check=(0, 1))
        b_dA += tl.dot(b_du, tl.trans(b_v_beta))
        b_dv_beta = tl.dot(b_A, b_du)
        b_dv = b_dv_beta * b_beta[:, None]
        b_dbeta += tl.sum(b_dv_beta * b_v, 1)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_dA = tl.where(m_A, b_dA, 0)
    b_dA = tl.dot(b_dA.to(b_A.dtype), b_A)
    b_dA = tl.dot(b_A, b_dA.to(b_A.dtype))
    b_dA = tl.where(m_A, -b_dA * exp2(b_g0[:, None] - b_g[None, :]), 0).to(k.dtype.element_ty)
    b_dA = b_dA.to(k.dtype.element_ty)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_p = tl.make_block_ptr(p + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dk = tl.make_block_ptr(dk + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dp = tl.make_block_ptr(dp + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_p = tl.load(p_p, boundary_check=(0, 1))
        b_dp = tl.load(p_dp, boundary_check=(0, 1))
        b_p_beta = (b_p * b_beta[:, None]).to(b_p.dtype)
        b_A += tl.dot(b_p_beta, tl.trans(b_k))
        b_dp_beta = tl.dot(b_dA, b_k)
        b_dbeta += tl.sum(b_dp_beta * b_p, 1)
        b_dk = tl.dot(tl.trans(b_dA), b_p_beta)
        b_dp += b_dp_beta * b_beta[:, None]
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dp, b_dp.to(p_dp.dtype.element_ty), boundary_check=(0, 1))

    b_dA_A = b_dA * b_A
    b_dg0 += tl.sum(b_dA_A, axis=1)
    b_dg = - tl.sum(b_dA_A, axis=0)
    p_dg = tl.make_block_ptr(dg + (bos*H + i_h), (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_dg0 = tl.make_block_ptr(dg0 + (bos*H + i_h), (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_dbeta = tl.make_block_ptr(dbeta + (bos*H + i_h), (T,), (H,), (i_t * BT,), (BT,), (0,))
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))
    tl.store(p_dg0, b_dg0.to(p_dg0.dtype.element_ty), boundary_check=(0,))
    tl.store(p_dbeta, b_dbeta.to(p_dbeta.dtype.element_ty), boundary_check=(0,))
