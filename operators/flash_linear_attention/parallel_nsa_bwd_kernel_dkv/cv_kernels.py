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
def parallel_nsa_bwd_kernel_dkv(q, k, v, lse, delta, do, dk, dv, block_mask, cu_seqlens, chunk_indices, scale, T, B: tl.constexpr, H: tl.constexpr, HQ: tl.constexpr, G: tl.constexpr, K: tl.constexpr, V: tl.constexpr, M: tl.constexpr, BS: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, IS_VARLEN: tl.constexpr):
    i_v, i_s, i_bh = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    i_b, i_h = (i_bh // H, i_bh % H)
    all = B * T
    if IS_VARLEN:
        i_n, i_s = (tl.load(chunk_indices + i_s * 2).to(tl.int32), tl.load(chunk_indices + i_s * 2 + 1).to(tl.int32))
        bos, eos = (tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32))
        T = eos - bos
    else:
        bos, eos = (i_b * T, i_b * T + T)
    p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_s * BS, 0), (BS, BK), (1, 0))
    p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_s * BS, i_v * BV), (BS, BV), (1, 0))
    p_dk = tl.make_block_ptr(dk + (i_v * all * H + bos * H + i_h) * K, (T, K), (H * K, 1), (i_s * BS, 0), (BS, BK), (1, 0))
    p_dv = tl.make_block_ptr(dv + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_s * BS, i_v * BV), (BS, BV), (1, 0))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_dk = tl.zeros([BS, BK], dtype=tl.float32)
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_dv = tl.zeros([BS, BV], dtype=tl.float32)
    for i in range(i_s * BS, T):
        b_m = tl.load(block_mask + (bos + i) * H * M + i_h * M + i_s)
        if b_m:
            p_q = tl.make_block_ptr(q + (bos + i) * HQ * K, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_q = (b_q * scale).to(b_q.dtype)
            p_do = tl.make_block_ptr(do + (bos + i) * HQ * V, (HQ, V), (V, 1), (i_h * G, i_v * BV), (G, BV), (1, 0))
            p_lse = lse + (bos + i) * HQ + i_h * G + tl.arange(0, G)
            p_delta = delta + (bos + i) * HQ + i_h * G + tl.arange(0, G)
            b_do = tl.load(p_do, boundary_check=(0, 1))
            b_lse = tl.load(p_lse)
            b_delta = tl.load(p_delta)
            b_s = tl.dot(b_k, tl.trans(b_q))
            b_p = exp(b_s - b_lse[None, :])
            b_p = tl.where((i >= i_s * BS + tl.arange(0, BS))[:, None], b_p, 0)
            b_dv += tl.dot(b_p.to(b_do.dtype), b_do)
            b_dp = tl.dot(b_v, tl.trans(b_do))
            b_ds = b_p * (b_dp - b_delta[None, :])
            b_dk += tl.dot(b_ds.to(b_q.dtype), b_q)
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
