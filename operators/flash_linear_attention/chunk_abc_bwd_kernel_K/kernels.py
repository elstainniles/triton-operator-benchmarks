# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: tests/ops/ascend_ops/cv_operator/chunk_abc/chunk_abc.py:738 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
# Source archive: dependencies and launch adaptation are not executable here.
# The complete original file and dependency snapshots are under upstream/.
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['T'])
def chunk_abc_bwd_kernel_K(
    q,
    k,
    v,
    z,
    h,
    A,
    do,
    dh,
    dq,
    dk,
    dv,
    dA,
    scale,
    T,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_p = tl.maximum(i_t * BT - 1, 0)
    n_bh = tl.num_programs(2)

    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] >= o_i[None, :]

    p_q = tl.make_block_ptr(q + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_A = tl.make_block_ptr(A + (i_k*n_bh+i_bh) * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_A = tl.dot((b_q * scale).to(b_q.dtype), tl.trans(b_k), allow_tf32=False)
    b_A = tl.where(m_s, b_A, 0.)
    tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))

    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + i_bh * T*V, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_z = tl.make_block_ptr(z + i_bh * T*V, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_zp = tl.make_block_ptr(z + i_bh * T*V, (T * V,), (1,), (i_p * V + i_v * BV,), (BV,), (0,))
        p_zc = tl.make_block_ptr(z + i_bh * T*V, (T * V,), (1,), ((i_t * BT + BT - 1) * V + i_v * BV,), (BV,), (0,))
        p_h = tl.make_block_ptr(h + i_bh * NT*K*V + i_t * K*V, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))

        p_do = tl.make_block_ptr(do + i_bh * T*V, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * NT*K*V + i_t * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv + (i_k*n_bh+i_bh) * T*V, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

        b_zp = tl.load(p_zp, boundary_check=(0,))
        b_zc = tl.load(p_zc, boundary_check=(0,))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_v = exp(b_v - b_zc[None, :]).to(b_v.dtype)
        b_z = tl.load(p_z, boundary_check=(0, 1))
        b_z = exp(b_zp[None, :] - b_z)
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_do = (b_do * b_z * scale).to(b_do.dtype)
        b_dh = tl.load(p_dh, boundary_check=(0, 1))

        b_dq += tl.dot(b_do, b_h, allow_tf32=False)
        b_dk += tl.dot(b_v, tl.trans(b_dh), allow_tf32=False)
        b_dv = b_v * tl.dot(b_k, b_dh, allow_tf32=False)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
    p_dA = tl.make_block_ptr(dA + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_dA = tl.load(p_dA, boundary_check=(0, 1))
    b_dq += tl.dot(b_dA, b_k, allow_tf32=False)
    b_dk += tl.dot(tl.trans(b_dA).to(b_k.dtype), b_q, allow_tf32=False)

    p_dq = tl.make_block_ptr(dq + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
