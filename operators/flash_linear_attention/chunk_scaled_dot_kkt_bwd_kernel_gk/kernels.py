# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: fla/ops/gated_oja_rule/chunk_kkt.py:231 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
# Source archive: dependencies and launch adaptation are not executable here.
# The complete original file and dependency snapshots are under upstream/.
import triton
import triton.language as tl


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8]
    ],
    key=['BK', 'NC', 'BT'],
)
@triton.jit(do_not_specialize=['B', 'T'])
def chunk_scaled_dot_kkt_bwd_kernel_gk(
    k,
    g,
    beta,
    dA,
    dk,
    dg,
    db,
    cu_seqlens,
    chunk_indices,
    B,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_k, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    i_t, i_i = i_c // NC, i_c % NC

    all = B * T
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    else:
        bos, eos = i_b * T, i_b * T + T
    T = eos - bos
    if i_t * BT + i_i * BC >= T:
        return

    o_k = i_k * BK + tl.arange(0, BK)
    m_k = o_k < K

    k += (bos * H + i_h) * K
    g += (bos * H + i_h) * K
    beta += bos * H + i_h

    dA += (bos * H + i_h) * BT
    dk += (bos * H + i_h) * K
    dg += (bos * H + i_h) * K
    db += (i_k * all + bos) * H + i_h

    p_g = tl.make_block_ptr(g, (T, K), (H*K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_b = tl.make_block_ptr(beta, (T,), (H,), (i_t * BT + i_i * BC,), (BC,), (0,))
    # [BC, BK]
    b_g = tl.load(p_g, boundary_check=(0, 1))
    b_dk = tl.zeros([BC, BK], dtype=tl.float32)
    # [BC]
    b_b = tl.load(p_b, boundary_check=(0,))
    if i_i > 0:
        p_gn = g + (i_t * BT + i_i * BC) * H*K + o_k
        # [BK,]
        b_gn = tl.load(p_gn, mask=m_k, other=0)
        for i_j in range(0, i_i):
            p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_gk = tl.make_block_ptr(g, (T, K), (H*K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_dA = tl.make_block_ptr(dA, (T, BT), (H*BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
            # [BC, BK]
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_gk = tl.load(p_gk, boundary_check=(0, 1))
            b_kg = b_k * exp(b_gn[None, :] - b_gk)
            # [BC, BC]
            b_dA = tl.load(p_dA, boundary_check=(0, 1))
            # [BC, BK]
            b_dkb = tl.dot(b_dA, b_kg) * exp(b_g - b_gn[None, :])
            b_dk += b_dkb

    o_i = tl.arange(0, BC)
    m_dA = (i_t * BT + i_i * BC + o_i) < T
    o_dA = (i_t * BT + i_i * BC + o_i) * H*BT + i_i * BC
    p_kj = k + (i_t * BT + i_i * BC) * H*K + o_k
    p_gkj = g + (i_t * BT + i_i * BC) * H*K + o_k

    p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        # [BC]
        b_dA = tl.load(dA + o_dA + j, mask=m_dA, other=0)
        # [BK]
        b_kj = tl.load(p_kj, mask=m_k, other=0).to(tl.float32)
        b_gkj = tl.load(p_gkj, mask=m_k, other=0).to(tl.float32)
        # [BC, BK]
        m_i = o_i[:, None] >= j
        # [BC, BK]
        b_dkb = tl.where(m_i, b_dA[:, None] * b_kj[None, :] * exp(b_g - b_gkj[None, :]), 0.)
        b_dk += b_dkb

        p_kj += H*K
        p_gkj += H*K
    b_db = tl.sum(b_dk * b_k, 1)
    b_dk *= b_b[:, None]
    p_db = tl.make_block_ptr(db, (T,), (H,), (i_t * BT + i_i * BC,), (BC,), (0,))
    tl.store(p_db, b_db.to(p_db.dtype.element_ty), boundary_check=(0,))

    tl.debug_barrier()
    # [BC, BK]
    b_dkt = tl.zeros([BC, BK], dtype=tl.float32)

    NC = min(NC, tl.cdiv(T - i_t * BT, BC))
    if i_i < NC - 1:
        p_gn = g + (min(i_t * BT + i_i * BC + BC, T) - 1) * H*K + o_k
        # [BK,]
        b_gn = tl.load(p_gn, mask=m_k, other=0)
        for i_j in range(i_i + 1, NC):
            p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_gk = tl.make_block_ptr(g, (T, K), (H*K, 1), (i_t * BT + i_j * BC, i_k*BK), (BC, BK), (1, 0))
            p_dA = tl.make_block_ptr(dA, (BT, T), (1, H*BT), (i_i * BC, i_t * BT + i_j * BC), (BC, BC), (0, 1))
            p_b = tl.make_block_ptr(beta, (T,), (H,), (i_t * BT + i_j * BC,), (BC,), (0,))

            o_j = i_t * BT + i_j * BC + o_i
            m_j = o_j < T
            # [BC]
            b_b = tl.load(p_b, boundary_check=(0,))
            # [BC, BK]
            b_kb = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32) * b_b[:, None]
            b_gk = tl.load(p_gk, boundary_check=(0, 1))
            b_kbg = b_kb * tl.where(m_j[:, None], exp(b_gk - b_gn[None, :]), 0)
            # [BC, BC]
            b_dA = tl.load(p_dA, boundary_check=(0, 1))
            # [BC, BK]
            # (SY 09/17) important to not use bf16 here to have a good precision.
            b_dkt += tl.dot(b_dA, b_kbg)
        b_dkt *= exp(b_gn[None, :] - b_g)
    o_dA = (i_t * BT + i_i * BC) * H*BT + i_i * BC + o_i
    p_kj = k + (i_t * BT + i_i * BC) * H*K + o_k
    p_gkj = g + (i_t * BT + i_i * BC) * H*K + o_k
    p_bj = beta + (i_t * BT + i_i * BC) * H

    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        # [BC,]
        b_dA = tl.load(dA + o_dA + j * H*BT)
        # [BK,]
        b_kbj = tl.load(p_kj, mask=m_k, other=0).to(tl.float32) * tl.load(p_bj)
        b_gkj = tl.load(p_gkj, mask=m_k, other=0).to(tl.float32)
        b_kbgj = b_kbj[None, :] * exp(b_gkj[None, :] - b_g)
        # [BC, BK]
        m_i = o_i[:, None] <= j
        b_dkt += tl.where(m_i, b_dA[:, None] * b_kbgj, 0.)

        p_kj += H*K
        p_gkj += H*K
        p_bj += H
    b_dg = (b_dk - b_dkt) * b_k
    b_dk += b_dkt

    p_dk = tl.make_block_ptr(dk, (T, K), (H*K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_dg = tl.make_block_ptr(dg, (T, K), (H*K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0, 1))
