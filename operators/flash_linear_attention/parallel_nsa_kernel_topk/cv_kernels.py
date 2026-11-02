# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Fixed CV launch adapter; JIT signatures and bodies match upstream/.
# Autotune/heuristics are bound explicitly by cases-cv.json.
# Generated from exact archived source; function bodies unchanged.
import triton
import triton.language as tl
import math

@triton.jit
def log(x):
    return tl.log(x.to(tl.float32))

@triton.jit
def _compare_and_swap(x, ids, flip, i: tl.constexpr, n_dims: tl.constexpr):
    n_outer: tl.constexpr = x.numel >> n_dims
    shape: tl.constexpr = [n_outer * 2 ** i, 2, 2 ** (n_dims - i - 1)]
    y = tl.reshape(x, shape)
    mask = tl.arange(0, 2)[None, :, None]
    left = tl.broadcast_to(tl.sum(y * (1 - mask), 1)[:, None, :], shape).to(y.dtype)
    right = tl.broadcast_to(tl.sum(y * mask, 1)[:, None, :], shape).to(y.dtype)
    left = tl.reshape(left, x.shape)
    right = tl.reshape(right, x.shape)
    y_idx = tl.reshape(ids, shape)
    left_idx = tl.broadcast_to(tl.sum(y_idx * (1 - mask), 1)[:, None, :], shape)
    right_idx = tl.broadcast_to(tl.sum(y_idx * mask, 1)[:, None, :], shape)
    left_idx = tl.reshape(left_idx, x.shape).to(y_idx.dtype)
    right_idx = tl.reshape(right_idx, x.shape).to(y_idx.dtype)
    idtype = tl.core.get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)
    ileft = left.to(idtype, bitcast=True)
    iright = right.to(idtype, bitcast=True)
    ix = x.to(idtype, bitcast=True)
    cond = (left > right) != flip
    ret = ix ^ tl.where(cond, ileft ^ iright, tl.zeros_like(ix))
    new_ids = ids ^ tl.where(cond, left_idx ^ right_idx, tl.zeros_like(ids))
    return (ret.to(x.dtype, bitcast=True), new_ids)

@triton.jit
def _bitonic_merge(x, ids, stage: tl.constexpr, order: tl.constexpr, n_dims: tl.constexpr):
    n_outer: tl.constexpr = x.numel >> n_dims
    tl.static_assert(stage <= n_dims)
    if order == 2:
        shape: tl.constexpr = [n_outer * 2 ** (n_dims - 1 - stage), 2, 2 ** stage]
        flip = tl.reshape(tl.broadcast_to(tl.arange(0, 2)[None, :, None], shape), x.shape)
    else:
        flip = order
    for i in tl.static_range(stage):
        x, ids = _compare_and_swap(x, ids, flip, i + (n_dims - stage), n_dims)
    return (x, ids)

@triton.jit
def exp(x):
    return tl.exp(x.to(tl.float32))

@triton.jit
def parallel_nsa_kernel_topk(q, k, lse, scale, block_indices, cu_seqlens, token_indices, chunk_offsets, T, H: tl.constexpr, HQ: tl.constexpr, G: tl.constexpr, K: tl.constexpr, S: tl.constexpr, BC: tl.constexpr, BS: tl.constexpr, BK: tl.constexpr, IS_VARLEN: tl.constexpr):
    i_t, i_bh = (tl.program_id(0), tl.program_id(1))
    i_b, i_h = (i_bh // H, i_bh % H)
    if IS_VARLEN:
        i_n, i_t = (tl.load(token_indices + i_t * 2).to(tl.int32), tl.load(token_indices + i_t * 2 + 1).to(tl.int32))
        bos, eos = (tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32))
        T = eos - bos
        boc = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = (i_b * T, i_b * T + T)
        boc = i_b * tl.cdiv(T, BS)
    p_q = tl.make_block_ptr(q + (bos + i_t) * HQ * K, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)
    TC = tl.cdiv(T, BS)
    NC = (i_t + 1) // BS
    if lse is not None:
        b_lse = tl.load(lse + (bos + i_t) * HQ + i_h * G + tl.arange(0, G))
    else:
        b_m = tl.full([G], float('-inf'), dtype=tl.float32)
        b_acc = tl.zeros([G], dtype=tl.float32)
        for i_c in range(0, NC, BC):
            o_c = i_c + tl.arange(0, BC)
            p_k = tl.make_block_ptr(k + (boc * H + i_h) * K, (K, TC), (1, H * K), (0, i_c), (BK, BC), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_s = tl.dot(b_q, b_k)
            b_s = tl.where((o_c < NC)[None, :], b_s, float('-inf'))
            b_m, b_mp = (tl.maximum(b_m, tl.max(b_s, 1)), b_m)
            b_r = exp(b_mp - b_m)
            b_p = exp(b_s - b_m[:, None])
            b_acc = b_acc * b_r + tl.sum(b_p, 1)
            b_mp = b_m
        if NC == 0:
            b_lse = tl.zeros([G], dtype=tl.float32)
        else:
            b_lse = b_m + log(b_acc)
    b_i = tl.full([BC], -1, dtype=tl.float32)
    o_i = tl.zeros([BC], dtype=tl.int32)
    m_i = tl.arange(0, BC) < BC // 2
    IC = i_t // BS
    for i_c in range(0, tl.cdiv(i_t + 1, BS), BC):
        o_c = i_c + tl.arange(0, BC)
        p_k = tl.make_block_ptr(k + (boc * H + i_h) * K, (K, TC), (1, H * K), (0, i_c), (BK, BC), (0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_s = tl.dot(b_q, b_k)
        b_s = tl.where(o_c < IC, b_s, float('-inf'))
        b_p = tl.where((o_c == 0) | ((o_c == IC - 1) | (o_c == IC)), 1.0, exp(b_s - b_lse[:, None]))
        b_i, b_ip = (tl.sum(b_p, 0), b_i)
        o_i, o_ip = (tl.where(o_c <= IC, o_c, -1), o_i)
        n_dims: tl.constexpr = tl.standard._log2(b_i.shape[0])
        for i in tl.static_range(1, n_dims):
            b_i, o_i = _bitonic_merge(b_i, o_i.to(tl.int32), i, 2, n_dims)
        if i_c != 0:
            b_i, o_i = _bitonic_merge(b_i, o_i.to(tl.int32), n_dims, False, n_dims)
            b_i_new = b_ip * m_i + b_i * (1 - m_i)
            o_i_new = o_ip * m_i + o_i * (1 - m_i)
            b_i, o_i = _bitonic_merge(b_i_new, o_i_new.to(tl.int32), n_dims, True, n_dims)
        else:
            b_i, o_i = _bitonic_merge(b_i, o_i.to(tl.int32), n_dims, True, n_dims)
    m_top = tl.arange(0, BC // S) == 0
    b_top = tl.sum(m_top[:, None] * tl.reshape(o_i, [BC // S, S]), 0)
    p_b = tl.make_block_ptr(block_indices + (bos + i_t) * H * S, (H * S,), (1,), (i_h * S,), (S,), (0,))
    tl.store(p_b, b_top.to(p_b.dtype.element_ty))
