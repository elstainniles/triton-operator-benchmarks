# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: tests/ops/ascend_ops/cv_operator/_parallel_rebased_bwd_dq_fp64/test__parallel_rebased_bwd_dq_fp64_npu.py:10 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
# Source archive: dependencies and launch adaptation are not executable here.
# The complete original file and dependency snapshots are under upstream/.
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['T'])
def _parallel_rebased_bwd_dq(
    i_bh,
    i_c,
    i_k,
    i_v,
    i_h,
    q,
    k,
    v,
    do,
    dz,
    dq,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BTL: tl.constexpr,
    BTS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    p_do = tl.make_block_ptr(do + i_bh * T*V, (T, V), (V, 1), (i_c*BTL, i_v*BV), (BTL, BV), (1, 0))
    p_q = tl.make_block_ptr(q + (i_bh) * T*K, (T, K), (K, 1), (i_c*BTL, i_k*BK), (BTL, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_do = tl.load(p_do, boundary_check=(0, 1)).to(b_q.dtype)
    b_q = (b_q * scale).to(b_q.dtype)
    b_dq = tl.zeros([BTL, BK], dtype=tl.float32)
    p_k = tl.make_block_ptr(k + i_bh * T*K, (T, K), (K, 1), (0, i_k*BK), (BTS, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * T*V, (V, T), (1, V), (i_v*BV, 0), (BV, BTS), (0, 1))
    p_dz = dz + i_bh * T + i_c*BTL + tl.arange(0, BTL)
    b_dz = tl.load(p_dz, mask=(i_c*BTL + tl.arange(0, BTL)) < T)

    for _ in range(0, i_c*BTL, BTS):
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_ds = tl.dot(b_do, b_v, allow_tf32=False)
        if i_v == 0:
            b_ds += b_dz[:, None]
        else:
            b_ds = b_ds
        b_s = tl.dot(b_q, tl.trans(b_k), allow_tf32=False)
        b_dq += tl.dot((2 * b_ds * b_s).to(b_v.dtype), b_k, allow_tf32=False)
        p_k = tl.advance(p_k, (BTS, 0))
        p_v = tl.advance(p_v, (0, BTS))

    b_dq *= scale
    o_q = tl.arange(0, BTL)
    o_k = tl.arange(0, BTS)
    p_k = tl.make_block_ptr(k + i_bh * T*K, (T, K), (K, 1), (i_c*BTL, i_k*BK), (BTS, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * T*V, (V, T), (1, V), (i_v*BV, i_c*BTL), (BV, BTS), (0, 1))
    for _ in range(i_c*BTL, (i_c + 1) * BTL, BTS):
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        m_s = o_q[:, None] >= o_k[None, :]
        b_ds = tl.dot(b_do, b_v, allow_tf32=False)
        if i_v == 0:
            b_ds += b_dz[:, None]
        else:
            b_ds = b_ds
        b_ds = tl.where(m_s, b_ds, 0) * scale
        b_s = tl.dot(b_q, tl.trans(b_k), allow_tf32=False)
        b_s = tl.where(m_s, b_s, 0)
        b_dq += tl.dot((2 * b_ds * b_s).to(b_k.dtype),
                       b_k, allow_tf32=False)
        p_k = tl.advance(p_k, (BTS, 0))
        p_v = tl.advance(p_v, (0, BTS))
        o_k += BTS
    p_dq = tl.make_block_ptr(dq + (i_bh + B * H * i_v) * T*K, (T, K), (K, 1), (i_c*BTL, i_k*BK), (BTL, BK), (1, 0))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    return
