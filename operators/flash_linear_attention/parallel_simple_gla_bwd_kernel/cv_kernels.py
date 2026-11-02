# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Fixed CV launch adapter; JIT signatures and bodies match upstream/.
# Autotune/heuristics are bound explicitly by cases-cv.json.
# Generated from exact archived source; function bodies unchanged.
import triton
import triton.language as tl
import math

@triton.jit(do_not_specialize=['T'])
def parallel_simple_gla_bwd_kernel_dkv(i_t, i_k, i_v, q, k, v, g, do, dk, dv, dg, scale, T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BS: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, USE_G: tl.constexpr):
    o_k = i_t * BT + tl.arange(0, BT)
    m_k = o_k < T
    p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    p_v = tl.make_block_ptr(v, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_dv = tl.zeros([BT, BV], dtype=tl.float32)
    if USE_G:
        b_gk = tl.load(g + o_k * H, mask=m_k, other=0)
    NTS = tl.cdiv(T, BS)
    for i_s in range(NTS * BS - BS, (i_t + 1) * BT - BS, -BS):
        p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_s, i_k * BK), (BS, BK), (1, 0))
        p_do = tl.make_block_ptr(do, (T, V), (H * V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
        o_q = i_s + tl.arange(0, BS)
        m_q = o_q < T
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_ds = tl.dot(b_v, tl.trans(b_do))
        b_s = tl.dot(b_k, tl.trans(b_q))
        if USE_G:
            b_gq = tl.load(g + o_q * H, mask=m_q, other=float('-inf'))
            b_gp = tl.load(g + (min(i_s + BS, T) - 1) * H)
            b_gn = tl.load(g + (i_s - 1) * H) if i_s % BT > 0 else 0.0
            if i_s >= 0:
                b_gpn = exp2(b_gp - b_gn)
                b_dk *= b_gpn
                b_dv *= b_gpn
                b_gqn = exp2(b_gq - b_gn)
                b_ds *= b_gqn[None, :]
                b_s *= b_gqn[None, :]
        b_dk += tl.dot(b_ds.to(b_q.dtype), b_q)
        b_dv += tl.dot(b_s.to(b_do.dtype), b_do)
    if USE_G:
        b_gn = tl.load(g + (min(i_t * BT + BT, T) - 1) * H)
        if i_t >= 0:
            b_gpn = exp2(b_gn - b_gk)[:, None]
            b_dk *= b_gpn
            b_dv *= b_gpn
    for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
        p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_s, i_k * BK), (BS, BK), (1, 0))
        p_do = tl.make_block_ptr(do, (T, V), (H * V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
        o_q = i_s + tl.arange(0, BS)
        m_q = o_q < T
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_s = tl.dot(b_k, tl.trans(b_q))
        b_ds = tl.dot(b_v, tl.trans(b_do))
        if USE_G:
            b_gq = tl.load(g + o_q * H, mask=m_q, other=float('-inf'))
            if i_s >= 0:
                b_gkq = exp2(-b_gk[:, None] + b_gq[None, :])
                b_ds *= b_gkq
                b_s *= b_gkq
        m_s = o_k[:, None] <= o_q[None, :]
        b_s = tl.where(m_s, b_s, 0)
        b_ds = tl.where(m_s, b_ds, 0)
        b_dk += tl.dot(b_ds.to(b_q.dtype), b_q)
        b_dv += tl.dot(b_s.to(b_do.dtype), b_do)
    b_dk *= scale
    b_dv *= scale
    p_dk = tl.make_block_ptr(dk, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dv = tl.make_block_ptr(dv, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
    if USE_G:
        b_dg = tl.load(dg + o_k * H, mask=m_k, other=0)
        b_dg -= tl.sum(b_dk * b_k, 1)
        tl.store(dg + o_k * H, b_dg.to(dg.dtype.element_ty), mask=m_k)

@triton.jit
def exp2(x):
    return tl.math.exp2(x.to(tl.float32))

@triton.jit(do_not_specialize=['T'])
def parallel_simple_gla_bwd_kernel_dq(i_t, i_k, i_v, q, k, v, g, do, dq, dg, scale, T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BS: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, USE_G: tl.constexpr):
    p_do = tl.make_block_ptr(do, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    b_do = tl.load(p_do, boundary_check=(0, 1))
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    o_q = i_t * BT + tl.arange(0, BT)
    m_q = o_q < T
    for i_s in range(0, i_t * BT, BS):
        p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_s, i_k * BK), (BS, BK), (1, 0))
        p_v = tl.make_block_ptr(v, (V, T), (1, H * V), (i_v * BV, i_s), (BV, BS), (0, 1))
        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_ds = tl.dot(b_do, b_v)
        if USE_G:
            b_g = tl.load(g + o_k * H, mask=m_k, other=0)
            b_gn = tl.load(g + (min(i_s + BS, T) - 1) * H)
            b_gp = tl.load(g + (i_s - 1) * H) if i_s % BT > 0 else 0.0
            b_ds *= tl.where(m_k, exp2(b_gn - b_g), 0)[None, :]
            if i_s > 0:
                b_dq *= exp2(b_gn - b_gp)
        b_dq += tl.dot(b_ds.to(b_v.dtype), b_k)
    if USE_G:
        b_gq = tl.load(g + o_q * H, mask=m_q, other=float('-inf'))
        b_dq *= exp2(b_gq)[:, None]
    for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
        p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_s, i_k * BK), (BS, BK), (1, 0))
        p_v = tl.make_block_ptr(v, (V, T), (1, H * V), (i_v * BV, i_s), (BV, BS), (0, 1))
        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_ds = tl.dot(b_do, b_v)
        if USE_G:
            b_gk = tl.load(g + o_k * H, mask=m_k, other=0)
            b_ds *= exp2(b_gq[:, None] - b_gk[None, :])
        m_s = (o_q[:, None] >= o_k[None, :]) & (m_q[:, None] & m_k[None, :])
        b_ds = tl.where(m_s, b_ds, 0)
        b_dq += tl.dot(b_ds.to(b_k.dtype), b_k)
    b_dq *= scale
    p_dq = tl.make_block_ptr(dq, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    if USE_G:
        p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_dg = tl.sum(b_dq * b_q, 1)
        p_dg = tl.make_block_ptr(dg, (T,), (H,), (i_t * BT,), (BT,), (0,))
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))

@triton.jit(do_not_specialize=['T'])
def parallel_simple_gla_bwd_kernel(q, k, v, g, do, dq, dk, dv, dg, scale, cu_seqlens, chunk_indices, T, B: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BS: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, NV: tl.constexpr, IS_VARLEN: tl.constexpr, USE_G: tl.constexpr):
    i_kv, i_t, i_bh = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    i_k, i_v = (i_kv // NV, i_kv % NV)
    i_b, i_h = (i_bh // H, i_bh % H)
    dq += i_v * B * H * T * K
    dk += i_v * B * H * T * K
    dv += i_k * B * H * T * V
    if USE_G:
        dg += i_kv * B * H * T
    if IS_VARLEN:
        i_n, i_t = (tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32))
        bos, eos = (tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32))
        T = eos - bos
    else:
        bos, eos = (i_b * T, i_b * T + T)
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    do += (bos * H + i_h) * V
    dq += (bos * H + i_h) * K
    dk += (bos * H + i_h) * K
    dv += (bos * H + i_h) * V
    if USE_G:
        g += bos * H + i_h
        dg += bos * H + i_h
    parallel_simple_gla_bwd_kernel_dq(i_t=i_t, i_k=i_k, i_v=i_v, q=q, k=k, v=v, g=g, do=do, dq=dq, dg=dg, scale=scale, T=T, H=H, K=K, V=V, BT=BT, BS=BS, BK=BK, BV=BV, USE_G=USE_G)
    tl.debug_barrier()
    parallel_simple_gla_bwd_kernel_dkv(i_t=i_t, i_k=i_k, i_v=i_v, q=q, k=k, v=v, g=g, do=do, dk=dk, dv=dv, dg=dg, scale=scale, T=T, H=H, K=K, V=V, BT=BT, BS=BS, BK=BK, BV=BV, USE_G=USE_G)
