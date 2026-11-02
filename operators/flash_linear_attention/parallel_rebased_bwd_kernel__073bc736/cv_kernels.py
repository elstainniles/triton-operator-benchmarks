# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Fixed CV launch adapter; JIT signatures and bodies match upstream/.
# Autotune/heuristics are bound explicitly by cases-cv.json.
# Generated from exact archived source; function bodies unchanged.
import triton
import triton.language as tl
import math

@triton.jit(do_not_specialize=['T'])
def _parallel_rebased_bwd_dkv(i_bh, i_c, i_k, i_v, i_h, q, k, v, do, dz, dk, dv, scale, T, B: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BTL: tl.constexpr, BTS: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr):
    p_k = tl.make_block_ptr(k + i_bh * T * K, (T, K), (K, 1), (i_c * BTL, i_k * BK), (BTL, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * T * V, (T, V), (V, 1), (i_c * BTL, i_v * BV), (BTL, BV), (1, 0))
    b_k, b_v = (tl.load(p_k, boundary_check=(0, 1)), tl.load(p_v, boundary_check=(0, 1)))
    b_dk, b_dv = (tl.zeros([BTL, BK], dtype=tl.float32), tl.zeros([BTL, BV], dtype=tl.float32))
    for i in range(tl.cdiv(T, BTS) * BTS - BTS, (i_c + 1) * BTL - BTS, -BTS):
        p_q = tl.make_block_ptr(q + i_bh * T * K, (K, T), (1, K), (i_k * BK, i), (BK, BTS), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * T * V, (V, T), (1, V), (i_v * BV, i), (BV, BTS), (0, 1))
        p_dz = dz + i_bh * T + i + tl.arange(0, BTS)
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1)).to(b_q.dtype)
        b_dz = tl.load(p_dz, mask=i + tl.arange(0, BTS) < T)
        b_s = tl.dot(b_k.to(b_q.dtype), b_q, allow_tf32=False) * scale
        b_s2 = b_s * b_s
        b_dv += tl.dot(b_s2.to(b_q.dtype), tl.trans(b_do), allow_tf32=False)
        b_ds = tl.dot(b_v, b_do, allow_tf32=False) * scale
        if i_v == 0:
            b_ds += b_dz[None, :] * scale
        else:
            b_ds = b_ds
        b_dk += tl.dot((2 * b_ds * b_s).to(b_q.dtype), tl.trans(b_q), allow_tf32=False)
    tl.debug_barrier()
    o_q, o_k = (tl.arange(0, BTS), tl.arange(0, BTL))
    for i in range(i_c * BTL, (i_c + 1) * BTL, BTS):
        p_q = tl.make_block_ptr(q + i_bh * T * K, (K, T), (1, K), (i_k * BK, i), (BK, BTS), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * T * V, (V, T), (1, V), (i_v * BV, i), (BV, BTS), (0, 1))
        p_dz = dz + i_bh * T + i + tl.arange(0, BTS)
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1)).to(b_q.dtype)
        b_dz = tl.load(p_dz, mask=i + tl.arange(0, BTS) < T)
        m_s = o_k[:, None] <= o_q[None, :]
        b_s = tl.dot(b_k, b_q, allow_tf32=False) * scale
        b_s2 = b_s * b_s
        b_s = tl.where(m_s, b_s, 0)
        b_s2 = tl.where(m_s, b_s2, 0)
        b_ds = tl.dot(b_v, b_do, allow_tf32=False)
        if i_v == 0:
            b_ds += b_dz[None, :]
        else:
            b_ds = b_ds
        b_ds = tl.where(m_s, b_ds, 0) * scale
        b_dv += tl.dot(b_s2.to(b_q.dtype), tl.trans(b_do), allow_tf32=False)
        b_dk += tl.dot((2 * b_ds * b_s).to(b_q.dtype), tl.trans(b_q), allow_tf32=False)
        o_q += BTS
    p_dk = tl.make_block_ptr(dk + (i_bh + B * H * i_v) * T * K, (T, K), (K, 1), (i_c * BTL, i_k * BK), (BTL, BK), (1, 0))
    p_dv = tl.make_block_ptr(dv + (i_bh + B * H * i_k) * T * V, (T, V), (V, 1), (i_c * BTL, i_v * BV), (BTL, BV), (1, 0))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
    return

@triton.jit(do_not_specialize=['T'])
def parallel_rebased_bwd_kernel(q, k, v, do, dz, dq, dk, dv, scale, T, B: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BTL: tl.constexpr, BTS: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr):
    i_kv, i_c, i_bh = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    NV = tl.cdiv(V, BV)
    i_k = i_kv // NV
    i_v = i_kv % NV
    i_h = i_bh % H
    tl.debug_barrier()
    _parallel_rebased_bwd_dkv(i_bh, i_c, i_k, i_v, i_h, q, k, v, do, dz, dk, dv, scale, B=B, H=H, T=T, K=K, V=V, BTL=BTL, BTS=BTS, BK=BK, BV=BV)
