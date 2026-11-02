# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Fixed CV launch adapter; JIT signatures and bodies match upstream/.
# Autotune/heuristics are bound explicitly by cases-cv.json.
# Generated from exact archived source; function bodies unchanged.
import triton
import triton.language as tl
import math

@triton.jit
def exp2(x):
    return tl.math.exp2(x.to(tl.float32))

@triton.jit(do_not_specialize=['T'])
def chunk_rwkv6_fwd_A_kernel_intra_sub_inter(q, k, gi, ge, A, cu_seqlens, chunk_indices, scale, T, H: tl.constexpr, K: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr, BK: tl.constexpr, NC: tl.constexpr, IS_VARLEN: tl.constexpr):
    i_t, i_c, i_bh = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    i_b, i_h = (i_bh // H, i_bh % H)
    i_i, i_j = (i_c // NC, i_c % NC)
    if IS_VARLEN:
        i_n, i_t = (tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32))
        bos, eos = (tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32))
        T = eos - bos
    else:
        bos, eos = (i_b * T, i_b * T + T)
    if i_t * BT + i_i * BC >= T:
        return
    if i_i <= i_j:
        return
    m_i = i_t * BT + i_i * BC + tl.arange(0, BC) < T
    b_A = tl.zeros([BC, BC], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        p_q = tl.make_block_ptr(q + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_gq = tl.make_block_ptr(ge + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (K, T), (1, H * K), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
        p_gk = tl.make_block_ptr(gi + (bos * H + i_h) * K, (K, T), (1, H * K), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
        p_gn = gi + (bos + i_t * BT + i_i * BC - 1) * H * K + i_h * K + o_k
        b_gn = tl.load(p_gn, mask=m_k, other=0)
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_gq = tl.where(m_i[:, None] & m_k, tl.load(p_gq, boundary_check=(0, 1)), float('-inf'))
        b_qg = b_q * exp2(b_gq - b_gn[None, :]) * scale
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_gk = tl.load(p_gk, boundary_check=(0, 1))
        b_kg = b_k * exp2(b_gn[:, None] - b_gk)
        b_A += tl.dot(b_qg, b_kg)
    p_A = tl.make_block_ptr(A + (bos * H + i_h) * BT, (T, BT), (H * BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
    tl.store(p_A, b_A.to(A.dtype.element_ty), boundary_check=(0, 1))
