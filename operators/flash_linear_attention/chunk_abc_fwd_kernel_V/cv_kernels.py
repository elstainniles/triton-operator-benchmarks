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
def chunk_abc_fwd_kernel_V(q, v, z, h, o, A, scale, T, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, NT: tl.constexpr):
    i_v, i_t, i_bh = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    i_p = tl.maximum(i_t * BT - 1, 0)
    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q + i_bh * T * K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_z = tl.make_block_ptr(z + i_bh * T * K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * NT * K * V + i_t * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        p_zp = tl.make_block_ptr(z + i_bh * T * K, (T * K,), (1,), (i_p * K + i_k * BK,), (BK,), (0,))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)
        b_z = tl.load(p_z, boundary_check=(0, 1))
        b_zp = tl.load(p_zp, boundary_check=(0,))
        b_q = (b_q * exp(b_zp[None, :] - b_z)).to(b_q.dtype)
        b_h = tl.load(p_h, boundary_check=(0, 1))
        if i_k >= 0:
            b_o += tl.dot(b_q, b_h, allow_tf32=False)
    p_v = tl.make_block_ptr(v + i_bh * T * V, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o + i_bh * T * V, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_A = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_o += tl.dot(b_A.to(b_v.dtype), b_v, allow_tf32=False)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
