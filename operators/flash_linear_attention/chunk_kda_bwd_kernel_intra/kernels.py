# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: fla/ops/kda/chunk_intra.py:381 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
# Source archive: dependencies and launch adaptation are not executable here.
# The complete original file and dependency snapshots are under upstream/.
import triton
import triton.language as tl


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@fla_cache_autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['BK', 'NC', 'BT', 'HV'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['B', 'T'])
def chunk_kda_bwd_kernel_intra(
    q,
    k,
    g,
    beta,
    dAqk,
    dAkk,
    dq,
    dq2,
    dk,
    dk2,
    dg,
    dg2,
    db,
    cu_seqlens,
    chunk_indices,
    B,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    SAFE_GATE: tl.constexpr,
    USE_GATHER: tl.constexpr,
):
    i_kc, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)
    i_k, i_i = i_kc // NC, i_kc % NC

    all = B * T
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    else:
        bos, eos = i_b * T, i_b * T + T
    T = eos - bos

    i_ti = i_t * BT + i_i * BC
    if i_ti >= T:
        return

    o_k = i_k * BK + tl.arange(0, BK)
    m_k = o_k < K

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    g += (bos * HV + i_hv) * K
    beta += bos * HV + i_hv

    dAqk += (bos * HV + i_hv) * BT
    dAkk += (bos * HV + i_hv) * BT
    dq += (bos * HV + i_hv) * K
    dq2 += (bos * HV + i_hv) * K
    dk += (bos * HV + i_hv) * K
    dk2 += (bos * HV + i_hv) * K
    dg += (bos * HV + i_hv) * K
    dg2 += (bos * HV + i_hv) * K
    db += (i_k * all + bos) * HV + i_hv

    p_g = tl.make_block_ptr(g, (T, K), (HV*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)

    p_b = tl.make_block_ptr(beta, (T,), (HV,), (i_ti,), (BC,), (0,))
    b_b = tl.load(p_b, boundary_check=(0,))

    b_dq2 = tl.zeros([BC, BK], dtype=tl.float32)
    b_dk2 = tl.zeros([BC, BK], dtype=tl.float32)
    if i_i > 0:
        p_gn = g + i_ti * HV*K + o_k
        # [BK,]
        b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)[None, :]
        for i_j in range(0, i_i):
            p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_gk = tl.make_block_ptr(g, (T, K), (HV*K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_dAqk = tl.make_block_ptr(dAqk, (T, BT), (HV*BT, 1), (i_ti, i_j * BC), (BC, BC), (1, 0))
            p_dAkk = tl.make_block_ptr(dAkk, (T, BT), (HV*BT, 1), (i_ti, i_j * BC), (BC, BC), (1, 0))
            # [BC, BK]
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_gk = tl.load(p_gk, boundary_check=(0, 1))
            b_kg = b_k * exp2(b_gn - b_gk)
            # [BC, BC]
            b_dAqk = tl.load(p_dAqk, boundary_check=(0, 1))
            b_dAkk = tl.load(p_dAkk, boundary_check=(0, 1))
            # [BC, BK]
            b_dq2 += tl.dot(b_dAqk, b_kg)
            b_dk2 += tl.dot(b_dAkk, b_kg)
        b_gqn = exp2(b_g - b_gn)
        b_dq2 *= b_gqn
        b_dk2 *= b_gqn

    o_i = tl.arange(0, BC)
    m_dA = (i_ti + o_i) < T
    o_dA = (i_ti + o_i) * HV*BT + i_i * BC
    p_kj = k + i_ti * H*K + o_k
    p_gkj = g + i_ti * HV*K + o_k

    p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))

    if SAFE_GATE:
        if USE_GATHER:
            b_gn = gather(b_g, tl.full([1, BK], min(BC//2, T - i_ti - 1), dtype=tl.int16), axis=0)
        else:
            p_gn = g + (i_ti + min(BC // 2, T - i_ti - 1)) * HV*K + o_k
            b_gn = tl.load(p_gn, mask=m_k, other=0)[None, :]

        p_dAqk = tl.make_block_ptr(dAqk, (T, BT), (HV*BT, 1), (i_ti, i_i * BC), (BC, BC), (1, 0))
        p_dAkk = tl.make_block_ptr(dAkk, (T, BT), (HV*BT, 1), (i_ti, i_i * BC), (BC, BC), (1, 0))
        b_dAqk_diag_qk = tl.load(p_dAqk, boundary_check=(0, 1)).to(tl.float32)
        b_dAkk_diag_qk = tl.load(p_dAkk, boundary_check=(0, 1)).to(tl.float32)

        m_i_diag_qk = (o_i[:, None] >= o_i[None, :]) & ((i_ti + o_i[:, None]) < T) & ((i_ti + o_i[None, :]) < T)
        m_j_diag_qk = (i_ti + o_i[:, None]) < T

        b_dAqk_diag_qk = tl.where(m_i_diag_qk, b_dAqk_diag_qk, 0.)
        b_dAkk_diag_qk = tl.where(m_i_diag_qk, b_dAkk_diag_qk, 0.)
        b_g_diag_qk = tl.where(m_j_diag_qk, b_g - b_gn, 0.)
        exp_b_g_diag_qk = tl.where(m_j_diag_qk, exp2(b_g_diag_qk), 0.)
        exp_neg_b_g_diag_qk = tl.where(m_j_diag_qk, exp2(-b_g_diag_qk), 0.)

        b_k_exp_diag_qk = b_k * exp_neg_b_g_diag_qk
        b_dq2 += tl.dot(b_dAqk_diag_qk, b_k_exp_diag_qk) * exp_b_g_diag_qk
        b_dk2 += tl.dot(b_dAkk_diag_qk, b_k_exp_diag_qk) * exp_b_g_diag_qk
    else:
        for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
            # [BC]
            b_dAqk = tl.load(dAqk + o_dA + j, mask=m_dA, other=0)
            b_dAkk = tl.load(dAkk + o_dA + j, mask=m_dA, other=0)
            # [BK]
            b_kj = tl.load(p_kj, mask=m_k, other=0).to(tl.float32)
            b_gkj = tl.load(p_gkj, mask=m_k, other=0).to(tl.float32)
            # [BC, BK]
            m_i = o_i[:, None] >= j
            # [BC, BK]
            b_gqk = exp2(b_g - b_gkj[None, :])
            b_dq2 += tl.where(m_i, b_dAqk[:, None] * b_kj[None, :] * b_gqk, 0.)
            b_dk2 += tl.where(m_i, b_dAkk[:, None] * b_kj[None, :] * b_gqk, 0.)

            p_kj += H*K
            p_gkj += HV*K

    b_db = tl.sum(b_dk2 * b_k, 1)
    b_dk2 *= b_b[:, None]

    p_dq = tl.make_block_ptr(dq, (T, K), (HV*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_dq2 = tl.make_block_ptr(dq2, (T, K), (HV*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_db = tl.make_block_ptr(db, (T,), (HV,), (i_ti,), (BC,), (0,))

    b_dg2 = b_q * b_dq2
    b_dq2 = b_dq2 + tl.load(p_dq, boundary_check=(0, 1))
    tl.store(p_dq2, b_dq2.to(p_dq2.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_db, b_db.to(p_db.dtype.element_ty), boundary_check=(0,))

    tl.debug_barrier()
    b_dkt = tl.zeros([BC, BK], dtype=tl.float32)

    NC = min(NC, tl.cdiv(T - i_t * BT, BC))
    if i_i < NC - 1:
        p_gn = g + (min(i_ti + BC, T) - 1) * HV*K + o_k
        # [BK,]
        b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)[None, :]
        for i_j in range(i_i + 1, NC):
            p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_t*BT+i_j*BC, i_k*BK), (BC, BK), (1, 0))
            p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_gk = tl.make_block_ptr(g, (T, K), (HV*K, 1), (i_t * BT + i_j * BC, i_k*BK), (BC, BK), (1, 0))
            p_b = tl.make_block_ptr(beta, (T,), (HV,), (i_t * BT + i_j * BC,), (BC,), (0,))
            p_dAqk = tl.make_block_ptr(dAqk, (BT, T), (1, HV*BT), (i_i * BC, i_t * BT + i_j * BC), (BC, BC), (0, 1))
            p_dAkk = tl.make_block_ptr(dAkk, (BT, T), (1, HV*BT), (i_i * BC, i_t * BT + i_j * BC), (BC, BC), (0, 1))
            # [BC]
            b_b = tl.load(p_b, boundary_check=(0,))
            # [BC, BK]
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_kb = tl.load(p_k, boundary_check=(0, 1)) * b_b[:, None]
            b_gk = tl.load(p_gk, boundary_check=(0, 1)).to(tl.float32)
            # [BC, BC]
            b_dAqk = tl.load(p_dAqk, boundary_check=(0, 1))
            b_dAkk = tl.load(p_dAkk, boundary_check=(0, 1))

            o_j = i_t * BT + i_j * BC + o_i
            m_j = o_j < T
            # [BC, BK]
            b_gkn = exp2(b_gk - b_gn)
            b_qg = b_q * tl.where(m_j[:, None], b_gkn, 0)
            b_kbg = b_kb * tl.where(m_j[:, None], b_gkn, 0)
            # [BC, BK]
            # (SY 09/17) important to not use bf16 here to have a good precision.
            b_dkt += tl.dot(b_dAqk, b_qg)
            b_dkt += tl.dot(b_dAkk, b_kbg)
        b_dkt *= exp2(b_gn - b_g)
    o_dA = i_ti * HV*BT + i_i * BC + o_i
    p_qj = q + i_ti * H*K + o_k
    p_kj = k + i_ti * H*K + o_k
    p_gkj = g + i_ti * HV*K + o_k
    p_bj = beta + i_ti * HV

    if SAFE_GATE:
        if USE_GATHER:
            b_gn = gather(b_g, tl.full([1, BK], min(BC//2, T - i_ti - 1), dtype=tl.int16), axis=0)
        else:
            p_gn = g + (i_ti + min(BC // 2, T - i_ti - 1)) * HV*K + o_k
            b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)[None, :]
        p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        p_b = tl.make_block_ptr(beta, (T,), (HV,), (i_ti,), (BC,), (0,))
        b_b = tl.load(p_b, boundary_check=(0,))

        p_dAqk = tl.make_block_ptr(dAqk, (BT, T), (1, HV*BT), (i_i * BC, i_ti), (BC, BC), (0, 1))
        p_dAkk = tl.make_block_ptr(dAkk, (BT, T), (1, HV*BT), (i_i * BC, i_ti), (BC, BC), (0, 1))
        b_dAqk_diag_kk = tl.load(p_dAqk, boundary_check=(0, 1)).to(tl.float32)
        b_dAkk_diag_kk = tl.load(p_dAkk, boundary_check=(0, 1)).to(tl.float32)

        m_i_diag_kk = (o_i[:, None] <= o_i[None, :]) & ((i_ti + o_i[:, None]) < T) & ((i_ti + o_i[None, :]) < T)
        m_j_diag_kk = (i_ti + o_i[:, None]) < T

        b_dAqk_diag_kk = tl.where(m_i_diag_kk, b_dAqk_diag_kk, 0.)
        b_dAkk_diag_kk = tl.where(m_i_diag_kk, b_dAkk_diag_kk, 0.)
        # ensure numerical stability
        b_g_diag_kk = tl.where(m_j_diag_kk, b_g - b_gn, 0.)
        exp_b_g_diag_kk = tl.where(m_j_diag_kk, exp2(b_g_diag_kk), 0.)
        exp_neg_b_g_diag_kk = tl.where(m_j_diag_kk, exp2(-b_g_diag_kk), 0.)

        b_q_exp = b_q * exp_b_g_diag_kk
        b_kb_exp = b_k * b_b[:, None] * exp_b_g_diag_kk

        b_dkt += tl.dot(b_dAqk_diag_kk, b_q_exp) * exp_neg_b_g_diag_kk
        b_dkt += tl.dot(b_dAkk_diag_kk, b_kb_exp) * exp_neg_b_g_diag_kk
    else:
        for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
            # [BC,]
            b_dAqk = tl.load(dAqk + o_dA + j * HV*BT)
            b_dAkk = tl.load(dAkk + o_dA + j * HV*BT)
            # [BK,]
            b_qj = tl.load(p_qj, mask=m_k, other=0).to(tl.float32)
            b_kbj = tl.load(p_kj, mask=m_k, other=0).to(tl.float32) * tl.load(p_bj)
            b_gkj = tl.load(p_gkj, mask=m_k, other=0).to(tl.float32)
            # [BC, BK]
            m_i = o_i[:, None] <= j
            b_gkq = exp2(b_gkj[None, :] - b_g)
            b_dkt += tl.where(m_i, b_dAqk[:, None] * b_qj[None, :] * b_gkq, 0.)
            b_dkt += tl.where(m_i, b_dAkk[:, None] * b_kbj[None, :] * b_gkq, 0.)

            p_qj += H*K
            p_kj += H*K
            p_gkj += HV*K
            p_bj += HV
    p_dk = tl.make_block_ptr(dk, (T, K), (HV*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_dk2 = tl.make_block_ptr(dk2, (T, K), (HV*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_dg = tl.make_block_ptr(dg, (T, K), (HV*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_dg2 = tl.make_block_ptr(dg2, (T, K), (HV*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))

    b_dg2 += (b_dk2 - b_dkt) * b_k + tl.load(p_dg, boundary_check=(0, 1))
    b_dk2 += tl.load(p_dk, boundary_check=(0, 1))
    b_dk2 += b_dkt

    tl.store(p_dk2, b_dk2.to(p_dk2.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dg2, b_dg2.to(p_dg2.dtype.element_ty), boundary_check=(0, 1))
