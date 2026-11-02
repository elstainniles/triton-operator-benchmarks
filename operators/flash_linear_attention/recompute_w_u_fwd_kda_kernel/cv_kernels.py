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
def recompute_w_u_fwd_kda_kernel(q, k, qg, kg, v, beta, w, u, A, gk, cu_seqlens, chunk_indices, T, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, STORE_QG: tl.constexpr, STORE_KG: tl.constexpr, IS_VARLEN: tl.constexpr):
    i_t, i_bh = (tl.program_id(0), tl.program_id(1))
    i_b, i_hv = (i_bh // HV, i_bh % HV)
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        i_n, i_t = (tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32))
        bos, eos = (tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32))
        T = eos - bos
    else:
        bos, eos = (i_b * T, i_b * T + T)
    k += (bos * H + i_h) * K
    v += (bos * HV + i_hv) * V
    u += (bos * HV + i_hv) * V
    w += (bos * HV + i_hv) * K
    gk += (bos * HV + i_hv) * K
    beta += bos * HV + i_hv
    A += (bos * HV + i_hv) * BT
    if STORE_QG:
        q += (bos * H + i_h) * K
        qg += (bos * HV + i_hv) * K
    if STORE_KG:
        kg += (bos * HV + i_hv) * K
    p_b = tl.make_block_ptr(beta, (T,), (HV,), (i_t * BT,), (BT,), (0,))
    b_b = tl.load(p_b, boundary_check=(0,))
    p_A = tl.make_block_ptr(A, (T, BT), (HV * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_A = tl.load(p_A, boundary_check=(0, 1))
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_u = tl.make_block_ptr(u, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))
    for i_k in range(tl.cdiv(K, BK)):
        p_w = tl.make_block_ptr(w, (T, K), (HV * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = b_k * b_b[:, None]
        p_gk = tl.make_block_ptr(gk, (T, K), (HV * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_gk = tl.load(p_gk, boundary_check=(0, 1)).to(tl.float32)
        b_kb *= exp2(b_gk)
        if STORE_QG:
            p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_qg = tl.make_block_ptr(qg, (T, K), (HV * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_qg = b_q * exp2(b_gk)
            tl.store(p_qg, b_qg.to(p_qg.dtype.element_ty), boundary_check=(0, 1))
        if STORE_KG:
            last_idx = min(i_t * BT + BT, T) - 1
            o_k = i_k * BK + tl.arange(0, BK)
            m_k = o_k < K
            b_gn = tl.load(gk + last_idx * HV * K + o_k, mask=m_k, other=0.0).to(tl.float32)
            b_kg = b_k * tl.where((i_t * BT + tl.arange(0, BT) < T)[:, None], exp2(b_gn[None, :] - b_gk), 0)
            p_kg = tl.make_block_ptr(kg, (T, K), (HV * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            tl.store(p_kg, b_kg.to(p_kg.dtype.element_ty), boundary_check=(0, 1))
        b_w = tl.dot(b_A, b_kb.to(b_k.dtype))
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))
