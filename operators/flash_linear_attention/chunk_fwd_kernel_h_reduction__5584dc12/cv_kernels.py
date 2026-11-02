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
def chunk_fwd_kernel_h_reduction(h, g, gk, gv, kvt, ht, cu_seqlens, chunk_offsets, T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, USE_G: tl.constexpr, USE_GK: tl.constexpr, USE_GV: tl.constexpr, STORE_FINAL_STATE: tl.constexpr, IS_VARLEN: tl.constexpr):
    i_k, i_v, i_nh = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    i_n, i_h = (i_nh // H, i_nh % H)
    if IS_VARLEN:
        bos, eos = (tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32))
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = (i_n * T, i_n * T + T)
        NT = tl.cdiv(T, BT)
        boh = i_n * NT
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    for i_t in range(NT):
        p_h = tl.make_block_ptr(h + ((boh + i_t) * H + i_h) * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h += tl.load(p_h, boundary_check=(0, 1)).to(tl.float32)
        if i_t > 0:
            tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))
        last_idx = min(i_t * BT + BT, T) - 1
        if USE_G:
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
            b_h *= exp(b_g_last)
        if USE_GK:
            p_gk_last = gk + (bos + last_idx) * H * K + i_h * K + i_k * BK + tl.arange(0, BK)
            b_gk_last = tl.load(p_gk_last, mask=i_k * BK + tl.arange(0, BK) < K, other=0.0)
            b_h *= exp(b_gk_last)[:, None]
        if USE_GV:
            p_gv_last = gv + (bos + last_idx) * H * V + i_h * V + i_v * BV + tl.arange(0, BV)
            b_gv_last = tl.load(p_gv_last, mask=i_v * BV + tl.arange(0, BV) < V, other=0.0)
            b_h *= exp(b_gv_last)[None, :]
    if STORE_FINAL_STATE:
        p_kvt = tl.make_block_ptr(kvt + i_nh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        p_ht = tl.make_block_ptr(ht + i_nh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h += tl.load(p_kvt, boundary_check=(0, 1)).to(tl.float32)
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
