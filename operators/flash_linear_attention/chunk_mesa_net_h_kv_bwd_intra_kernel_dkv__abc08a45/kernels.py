# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: tests/ops/ascend_ops/cv_operator/chunk_mesa_net_h_kv_bwd_intra_kernel_dq/test_chunk_mesa_net_h_kv_bwd_intra_kernel_dq_npu.py:23 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
# Source archive: dependencies and launch adaptation are not executable here.
# The complete original file and dependency snapshots are under upstream/.
import triton
import triton.language as tl


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def chunk_mesa_net_h_kv_bwd_intra_kernel_dkv(
    q_star, k, v, beta, h_kv, g, do, dh_kv, dk_beta, dg, dv,
    cu_seqlens, chunk_indices,
    B: tl.constexpr, T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, NV: tl.constexpr, IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
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

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T

    v += (bos * H + i_h) * V
    do += (bos * H + i_h) * V
    h_kv += (i_tg * H + i_h).to(tl.int64) * K*V
    dh_kv += (i_tg * H + i_h).to(tl.int64) * K*V
    q_star += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    beta += (bos * H + i_h)
    g += bos * H + i_h
    dg += bos * H + i_h
    dk_beta += (bos * H + i_h) * K
    dv += (bos * H + i_h) * V

    p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_beta = tl.make_block_ptr(beta, (T, ), (H, ), (i_t * BT,), (BT,), (0,))
    p_q = tl.make_block_ptr(q_star, (T, K), (H*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_beta = tl.load(p_beta, boundary_check=(0, ))
    b_g = tl.load(p_g, boundary_check=(0,))
    b_g_last = tl.load(g + (min(i_t * BT + BT, T) - 1) * H)

    b_m = tl.where((o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t[None, :]), exp(b_g[:, None] - b_g[None, :]), 0)
    b_k = (b_k * b_beta[:, None]).to(b_k.dtype)
    b_s = tl.dot(b_q, tl.trans(b_k)) * b_m
    b_g_exp_k = tl.where(m_t, exp(-b_g + b_g_last), 0)

    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_dg_last = tl.zeros([1], dtype=tl.float32)
    b_dg = tl.zeros([BT], dtype=tl.float32)

    for i_v in tl.static_range(0, NV):
        v_off = i_v * BV
        p_v = tl.make_block_ptr(v, (T, V), (H*V, 1), (i_t * BT, v_off), (BT, BV), (1, 0))
        p_do = tl.make_block_ptr(do, (T, V), (H*V, 1), (i_t * BT, v_off), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h_kv, (V, K), (1, V), (v_off, 0), (BV, BK), (0, 1))
        p_dh = tl.make_block_ptr(dh_kv, (V, K), (1, V), (v_off, 0), (BV, BK), (0, 1))

        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_dh = tl.load(p_dh, boundary_check=(0, 1))

        b_dg_last += tl.sum(b_h * b_dh) * exp(b_g_last)

        b_ds = tl.dot(b_do, tl.trans(b_v))
        b_dm = b_s * b_ds
        b_dm = tl.where(tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :], b_dm, 0)
        b_dg += tl.sum(b_dm, axis=1)
        b_dg -= tl.sum(b_dm, axis=0)

        b_ds = b_ds * b_m
        b_dk_inter = tl.dot(b_v, b_dh.to(b_v.dtype)) * b_g_exp_k[:, None]
        b_dk += b_dk_inter
        b_dg_last += tl.sum(b_dk_inter * b_k)
        b_dg -= tl.sum(b_dk_inter * b_k, axis=1)

        b_dv = tl.dot(b_k, tl.trans(b_dh).to(b_k.dtype)) * b_g_exp_k[:, None] + tl.dot(tl.trans(b_s.to(b_do.dtype)), b_do)
        b_dk += tl.dot(tl.trans(b_ds.to(b_q.dtype)), b_q)

        p_dv = tl.make_block_ptr(dv, (T, V), (H*V, 1), (i_t * BT, v_off), (BT, BV), (1, 0))
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))

    b_dg = tl.where(o_t < min(i_t * BT + BT, T) - 1, b_dg, b_dg + b_dg_last)
    p_dk = tl.make_block_ptr(dk_beta, (T, K), (H*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_dg = tl.make_block_ptr(dg, (T,), (H,), (i_t * BT,), (BT,), (0,))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))
