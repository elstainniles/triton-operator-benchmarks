# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Fixed CV launch adapter; JIT signatures and bodies match upstream/.
# Autotune/heuristics are bound explicitly by cases-cv.json.
# Generated from exact archived source; function bodies unchanged.
import triton
import triton.language as tl
import math

@triton.jit
def chunk_dplr_bwd_kernel_dv(A_qk, kg, do, dv, dh, cu_seqlens, chunk_indices, T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, IS_VARLEN: tl.constexpr):
    i_v, i_t, i_bh = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    i_b, i_h = (i_bh // H, i_bh % H)
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = (tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32))
        bos, eos = (tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32))
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = (i_b * T, i_b * T + T)
    b_dv = tl.zeros([BT, BV], dtype=tl.float32)
    A_qk += (bos * H + i_h) * BT
    do += (bos * H + i_h) * V
    dv += (bos * H + i_h) * V
    kg += (bos * H + i_h) * K
    dh += (i_tg * H + i_h) * K * V
    stride_qk = H * K
    stride_vo = H * V
    stride_A = H * BT
    for i_k in range(tl.cdiv(K, BK)):
        p_dh = tl.make_block_ptr(dh, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        p_kg = tl.make_block_ptr(kg, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_dh = tl.load(p_dh, boundary_check=(0, 1))
        b_kg = tl.load(p_kg, boundary_check=(0, 1))
        b_dv += tl.dot(b_kg, b_dh.to(b_kg.dtype))
    p_Aqk = tl.make_block_ptr(A_qk, (BT, T), (1, stride_A), (0, i_t * BT), (BT, BT), (0, 1))
    b_A = tl.where(tl.arange(0, BT)[:, None] <= tl.arange(0, BT)[None, :], tl.load(p_Aqk, boundary_check=(0, 1)), 0)
    p_do = tl.make_block_ptr(do, (T, V), (stride_vo, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_dv = tl.make_block_ptr(dv, (T, V), (stride_vo, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    b_do = tl.load(p_do, boundary_check=(0, 1))
    b_dv += tl.dot(b_A.to(b_do.dtype), b_do)
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
