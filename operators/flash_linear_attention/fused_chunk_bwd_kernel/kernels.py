# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: tests/ops/ascend_ops/cv_operator/fused_chunk_bwd_kernel_fp64/test_fused_chunk_bwd_kernel_fp64_npu.py:43 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
# Source archive: dependencies and launch adaptation are not executable here.
# The complete original file and dependency snapshots are under upstream/.
import triton
import triton.language as tl


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_G_GAMMA': lambda args: args['g_gamma'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    'USE_INITIAL_STATE': lambda args: args['dh0'] is not None,
    'USE_FINAL_STATE': lambda args: args['dht'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS
        for num_stages in [1]
    ],
    key=['H', 'K', 'V', 'BT'],
)
@triton.jit(do_not_specialize=['T'])
def fused_chunk_bwd_kernel(
    q,
    k,
    v,
    g,
    g_gamma,
    do,
    dq,
    dk,
    dv,
    dg,
    h0,
    dht,
    dh0,
    cu_seqlens,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE: tl.constexpr,
):
    i_v, i_k, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_h = i_nh // H, i_nh % H

    all = B * T
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
    NT = tl.cdiv(T, BT)
    NV = tl.cdiv(V, BV)

    o_i = tl.arange(0, BT)
    if USE_G_GAMMA:
        b_gamma = tl.load(g_gamma + i_h)
        b_g = b_gamma * (o_i + 1)
        b_g_last = b_gamma * BT
        b_gq = exp(b_g)
        b_gk = exp(b_g_last - b_g)
        b_gn = exp(b_g_last)

    m_s = o_i[:, None] >= o_i[None, :]

    q = q + (bos*H + i_h) * K
    k = k + (bos*H + i_h) * K
    v = v + (bos*H + i_h) * V
    do = do + (bos*H + i_h) * V
    dq = dq + (i_v * all + bos).to(tl.int64) * H*K + i_h * K
    dk = dk + (i_v * all + bos).to(tl.int64) * H*K + i_h * K
    dv = dv + (i_k * all + bos).to(tl.int64) * H*V + i_h * V

    # [BV, BK]
    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h = tl.make_block_ptr(h0 + i_nh * K*V, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1)).to(tl.float32)

    for i_t in range(0, NT):
        p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_v = tl.make_block_ptr(v, (V, T), (1, H*V), (i_v * BV, i_t * BT), (BV, BT), (0, 1))
        p_do = tl.make_block_ptr(do, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dq = tl.make_block_ptr(dq, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))

        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        # [BT, BK]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BV, BT]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BT, BV]
        b_do = tl.load(p_do, boundary_check=(0, 1))
        last_idx = min(i_t * BT + BT, T) - 1

        # [BT, BT]
        b_ds = tl.dot(b_do, b_v) * scale

        # scalar decay
        if USE_G:
            p_g = g + (bos + o_t) * H + i_h
            b_g = tl.load(p_g, mask=(o_t < T), other=0.)
            b_g_last = tl.load(g + (bos + last_idx) * H + i_h)

            b_gq = exp(b_g)
            b_gk = exp(b_g_last - b_g)
            b_gn = exp(b_g_last)

            p_dg = dg + ((i_k * NV + i_v) * all + (bos + o_t)).to(tl.int64) * H + i_h
            # [BT, BT]
            b_gs = tl.where(m_s & m_t, exp(b_g[:, None] - b_g[None, :]), 0)
            b_ds = b_ds * b_gs
            # [BT, BK]
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_dq = tl.dot(b_ds.to(b_k.dtype), b_k) + tl.dot((b_do * b_gq[:, None] * scale).to(b_k.dtype), b_h.to(b_k.dtype))
            # [BT]
            b_dg_t = tl.sum(b_q * b_dq, 1)
            tl.store(p_dg, b_dg_t.to(p_dg.dtype.element_ty), mask=m_t)
            # [BV, BK]
            b_h = b_h * b_gn + tl.dot(b_v, (b_k * b_gk[:, None]).to(b_k.dtype))

        elif USE_G_GAMMA:
            b_g_last = b_gamma * min(BT, T - i_t * BT)
            b_gk = exp(b_g_last - b_g)
            b_gn = exp(b_g_last)

            # [BT, BT]
            b_gs = tl.where(m_s & m_t, exp(b_g[:, None] - b_g[None, :]), 0)
            b_ds = b_ds * b_gs
            # [BT, BK]
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_dq = tl.dot(b_ds.to(b_k.dtype), b_k) + tl.dot((b_do * b_gq[:, None] * scale).to(b_k.dtype), b_h.to(b_k.dtype))
            # [BV, BK]
            b_h = b_h * b_gn + tl.dot(b_v, (b_k * b_gk[:, None]).to(b_k.dtype))

        else:
            # [BT, BT]
            b_ds *= m_s & m_t
            # [BT, BK]
            b_dq = tl.dot(b_ds.to(b_k.dtype), b_k) + tl.dot((b_do * scale).to(b_k.dtype), b_h.to(b_k.dtype))
            # [BV, BK]
            b_h += tl.dot(b_v, b_k)

        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))

    # [BK, BV]
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_FINAL_STATE:
        p_dh = tl.make_block_ptr(dht + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_dh += tl.load(p_dh, boundary_check=(0, 1)).to(tl.float32)

    if USE_G:
        b_dg = tl.zeros([BT], dtype=tl.float32)
        b_dg_last = tl.sum(tl.trans(b_h) * b_dh)

    # sync threads
    b_h = None
    tl.debug_barrier()

    for i_t in range(NT - 1, -1, -1):
        p_q = tl.make_block_ptr(q, (K, T), (1, H*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_v = tl.make_block_ptr(v, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_do = tl.make_block_ptr(do, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dk = tl.make_block_ptr(dk, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dv = tl.make_block_ptr(dv, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        # [BK, BT]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BT, BK]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        last_idx = min(i_t * BT + BT, T) - 1

        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        # [BT, BT]
        b_s = tl.dot(b_k, b_q)
        b_ds = tl.dot(b_v, tl.trans(b_do))

        if USE_G:
            p_g = g + (bos + o_t) * H + i_h
            p_dg = dg + ((i_k * NV + i_v) * all + (bos + o_t)).to(tl.int64) * H + i_h
            b_g = tl.load(p_g, mask=m_t, other=0.)
            b_g_last = tl.load(g + (bos + last_idx) * H + i_h)

            b_gq = exp(b_g)
            b_gk = exp(b_g_last - b_g)
            b_gn = exp(b_g_last)
            b_gs = tl.trans(tl.where(m_s & (m_t[:, None] & m_t), exp(b_g[:, None] - b_g[None, :]), 0)) * scale

            b_s = b_s * b_gs
            b_ds = b_ds * b_gs

            # [BT, BK]
            b_dk = tl.dot(b_ds.to(b_k.dtype), tl.trans(b_q)) + tl.dot(b_v, tl.trans(b_dh).to(b_v.dtype)) * b_gk[:, None]

            # [BT]
            b_dg_t = tl.where(m_t, tl.load(p_dg, mask=m_t, other=0.) - tl.sum(b_k * b_dk, 1), 0)
            b_dg_last += tl.sum(b_dg_t, 0)
            b_dg = b_dg_last + b_dg_t - tl.cumsum(b_dg_t, 0)

            # [BT, BV]
            b_dv = tl.dot(b_s.to(b_do.dtype), b_do) + tl.dot(b_k, b_dh.to(b_k.dtype)) * b_gk[:, None]
            # [BK, BV]
            b_dh = b_dh * b_gn + tl.dot(b_q, (b_do * b_gq[:, None] * scale).to(b_do.dtype))

            tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), mask=m_t)

        elif USE_G_GAMMA:
            b_g_last = b_gamma * min(BT, T - i_t * BT)
            b_gk = exp(b_g_last - b_g)
            b_gn = exp(b_g_last)
            b_gs = tl.trans(tl.where(m_s & (m_t[:, None] & m_t), exp(b_g[:, None] - b_g[None, :]), 0)) * scale

            b_s = b_s * b_gs
            b_ds = b_ds * b_gs

            b_dk = tl.dot(b_ds.to(b_k.dtype), tl.trans(b_q)) + tl.dot(b_v, tl.trans(b_dh).to(b_v.dtype)) * b_gk[:, None]
            # [BT, BV]
            b_dv = tl.dot(b_s.to(b_do.dtype), b_do) + tl.dot(b_k, b_dh.to(b_k.dtype)) * b_gk[:, None]
            # [BK, BV]
            b_dh = b_dh * b_gn + tl.dot(b_q, (b_do * b_gq[:, None] * scale).to(b_do.dtype))

        else:
            mask = tl.trans(m_s & (m_t[:, None] & m_t))
            b_s = tl.where(mask, b_s * scale, 0).to(b_do.dtype)
            b_ds = tl.where(mask, b_ds * scale, 0).to(b_q.dtype)

            b_dk = tl.dot(b_ds, tl.trans(b_q)) + tl.dot(b_v, tl.trans(b_dh).to(b_v.dtype))
            # [BT, BV]
            b_dv = tl.dot(b_s.to(b_do.dtype), b_do) + tl.dot(b_k, b_dh.to(b_k.dtype))
            # [BK, BV]
            b_dh += tl.dot(b_q, (b_do * scale).to(b_do.dtype))

        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))

    if USE_INITIAL_STATE:
        p_dh0 = tl.make_block_ptr(dh0 + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), boundary_check=(0, 1))
