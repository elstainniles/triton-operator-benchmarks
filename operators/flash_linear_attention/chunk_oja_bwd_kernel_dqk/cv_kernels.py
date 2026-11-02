# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Fixed CV launch adapter; JIT signatures and bodies match upstream/.
# Autotune/heuristics are bound explicitly by cases-cv.json.
# Generated from exact archived source; function bodies unchanged.
import triton
import triton.language as tl
import math

@triton.jit
def exp(x):
    return tl.exp(x.to(tl.float32))

@triton.jit(do_not_specialize=['T'])
def chunk_oja_bwd_kernel_dqk(q, k, h, gv, A, dq, dk, dA, do, scale, cu_seqlens, chunk_indices, B, T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, IS_VARLEN: tl.constexpr):
    i_k, i_t, i_bh = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    i_b, i_h = (i_bh // H, i_bh % H)
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = (tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32))
        bos, eos = (tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32))
        all = T
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = (i_b * T, i_b * T + T)
        all = B * T
    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] >= o_i[None, :]
    p_q = tl.make_block_ptr(q + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_A = tl.make_block_ptr(A + ((i_k * all + bos) * H + i_h) * BT, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_A = tl.dot((b_q * scale).to(b_q.dtype), tl.trans(b_k))
    b_A = tl.where(m_s, b_A, 0.0)
    tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        p_h = tl.make_block_ptr(h + (i_tg * H + i_h) * K * V, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        p_do = tl.make_block_ptr(do + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_gv = tl.make_block_ptr(gv + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_gv = tl.load(p_gv, boundary_check=(0, 1))
        b_do = (b_do * exp(b_gv) * scale).to(b_do.dtype)
        b_dq += tl.dot(b_do, b_h.to(b_do.dtype))
    p_dA = tl.make_block_ptr(dA + (bos * H + i_h) * BT, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    p_dq = tl.make_block_ptr(dq + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    b_dA = tl.load(p_dA, boundary_check=(0, 1))
    b_dq += tl.dot(b_dA.to(b_q.dtype), b_k)
    b_dk = tl.dot(tl.trans(b_dA).to(b_q.dtype), b_q)
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
