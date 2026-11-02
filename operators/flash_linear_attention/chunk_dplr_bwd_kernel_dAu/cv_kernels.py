# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Fixed CV launch adapter; JIT signatures and bodies match upstream/.
# Autotune/heuristics are bound explicitly by cases-cv.json.
# Generated from exact archived source; function bodies unchanged.
import triton
import triton.language as tl
import math

@triton.jit(do_not_specialize=['T'])
def chunk_dplr_bwd_kernel_dAu(v, do, v_new, A_qb, dA_qk, dA_qb, dv_new, cu_seqlens, chunk_indices, scale: tl.constexpr, T, H: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BV: tl.constexpr, IS_VARLEN: tl.constexpr):
    i_t, i_bh = (tl.program_id(0), tl.program_id(1))
    i_b, i_h = (i_bh // H, i_bh % H)
    if IS_VARLEN:
        i_n, i_t = (tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32))
        bos, eos = (tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32))
    else:
        bos, eos = (i_b * T, i_b * T + T)
    T = eos - bos
    b_dA_qk = tl.zeros([BT, BT], dtype=tl.float32)
    b_dA_qb = tl.zeros([BT, BT], dtype=tl.float32)
    p_A_qb = tl.make_block_ptr(A_qb + (bos * H + i_h) * BT, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_A_qb = tl.load(p_A_qb, boundary_check=(0, 1))
    b_A_qb = tl.where(tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :], b_A_qb, 0.0).to(b_A_qb.dtype)
    for i_v in range(tl.cdiv(V, BV)):
        p_do = tl.make_block_ptr(do + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (V, T), (1, H * V), (i_v * BV, i_t * BT), (BV, BT), (0, 1))
        p_v_new = tl.make_block_ptr(v_new + (bos * H + i_h) * V, (V, T), (1, H * V), (i_v * BV, i_t * BT), (BV, BT), (0, 1))
        p_dv_new = tl.make_block_ptr(dv_new + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_v_new = tl.load(p_v_new, boundary_check=(0, 1))
        b_dA_qk += tl.dot(b_do, b_v)
        b_dA_qb += tl.dot(b_do, b_v_new)
        b_dv_new = tl.dot(tl.trans(b_A_qb), b_do)
        tl.store(p_dv_new, b_dv_new.to(p_dv_new.dtype.element_ty), boundary_check=(0, 1))
    p_dA_qk = tl.make_block_ptr(dA_qk + (bos * H + i_h) * BT, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    p_dA_qb = tl.make_block_ptr(dA_qb + (bos * H + i_h) * BT, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]
    b_dA_qk = tl.where(m_s, b_dA_qk * scale, 0.0)
    tl.store(p_dA_qk, b_dA_qk.to(p_dA_qk.dtype.element_ty), boundary_check=(0, 1))
    b_dA_qb = tl.where(m_s, b_dA_qb * scale, 0.0)
    tl.store(p_dA_qb, b_dA_qb.to(p_dA_qb.dtype.element_ty), boundary_check=(0, 1))
