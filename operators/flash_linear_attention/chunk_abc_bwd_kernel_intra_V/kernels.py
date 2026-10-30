# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: tests/ops/ascend_ops/cv_operator/chunk_abc/chunk_abc.py:611 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
# Source archive: dependencies and launch adaptation are not executable here.
# The complete original file and dependency snapshots are under upstream/.
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['T'])
def chunk_abc_bwd_kernel_intra_V(
    q,
    k,
    z,
    dA,
    dq,
    dk,
    T,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
):
    i_k, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_t, i_i = i_c // NC, i_c % NC

    p_z = tl.make_block_ptr(z + i_bh * T*K, (T, K), (K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_zn = tl.make_block_ptr(z + i_bh * T*K, (T * K,), (1,), ((i_t * BT + i_i * BC) * K + i_k * BK,), (BK,), (0,))
    b_zn = tl.load(p_zn, boundary_check=(0,))
    b_z = tl.load(p_z, boundary_check=(0, 1))
    b_zq = exp(b_zn[None, :] - b_z)
    b_dq = tl.zeros([BC, BK], dtype=tl.float32)
    for i_j in range(0, i_i):
        p_k = tl.make_block_ptr(k + i_bh * T*K, (T, K), (K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
        p_dA = tl.make_block_ptr(dA + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kz = exp(b_k - b_zn[None, :]).to(b_k.dtype)
        b_dA = tl.load(p_dA, boundary_check=(0, 1))
        b_dq += tl.dot(b_dA, b_kz, allow_tf32=False)
    b_dq *= b_zq

    o_i = tl.arange(0, BC)
    o_dA = i_bh * T * BT + (i_t * BT + i_i * BC + tl.arange(0, BC)) * BT + i_i * BC
    m_dA = (i_t * BT + i_i * BC + tl.arange(0, BC)) < T
    for j in range(0, BC):
        p_kj = tl.make_block_ptr(k + i_bh * T*K, (T * K,), (1,), ((i_t * BT + i_i*BC+j) * K + i_k * BK,), (BK,), (0,))
        b_dA = tl.load(dA + o_dA + j, mask=m_dA, other=0)
        b_kj = tl.load(p_kj, boundary_check=(0,)).to(tl.float32)
        m_i = o_i[:, None] >= j
        b_dq += tl.where(m_i, b_dA[:, None] * exp(b_kj[None, :] - b_z), 0.)
    p_dq = tl.make_block_ptr(dq + i_bh * T*K, (T, K), (K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))

    tl.debug_barrier()
    p_k = tl.make_block_ptr(k + i_bh * T*K, (T, K), (K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_zn = tl.make_block_ptr(z + i_bh * T*K, (T*K,), (1,), ((i_t * BT + i_i * BC + BC - 1) * K + i_k * BK,), (BK,), (0,))
    b_zn = tl.load(p_zn, boundary_check=(0,))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_kz = exp(b_k - b_zn[None, :])
    b_dk = tl.zeros([BC, BK], dtype=tl.float32)
    for i_j in range(i_i + 1, NC):
        p_q = tl.make_block_ptr(q + i_bh * T*K, (T, K), (K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
        p_z = tl.make_block_ptr(z + i_bh * T*K, (T, K), (K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
        p_dA = tl.make_block_ptr(dA + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT + i_j * BC, i_i * BC), (BC, BC), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_z = tl.load(p_z, boundary_check=(0, 1))
        b_qz = (b_q * exp(b_zn[None, :] - b_z)).to(b_q.dtype)
        b_dA = tl.load(p_dA, boundary_check=(0, 1))
        b_dk += tl.dot(tl.trans(b_dA), b_qz, allow_tf32=False)
    b_dk *= b_kz

    o_dA = i_bh * T * BT + (i_t * BT + i_i * BC) * BT + i_i * BC + tl.arange(0, BC)
    for j in range(0, BC):
        p_qj = tl.make_block_ptr(q + i_bh * T*K, (T * K,), (1,), ((i_t * BT + i_i * BC + j) * K + i_k * BK,), (BK,), (0,))
        p_zj = tl.make_block_ptr(z + i_bh * T*K, (T * K,), (1,), ((i_t * BT + i_i * BC + j) * K + i_k * BK,), (BK,), (0,))
        b_dA = tl.load(dA + o_dA + j * BT, mask=(i_t * BT + i_i * BC + j < T), other=0)
        b_qj = tl.load(p_qj, boundary_check=(0,)).to(tl.float32)
        b_zj = tl.load(p_zj, boundary_check=(0,)).to(tl.float32)
        m_i = o_i[:, None] <= j
        b_dk += tl.where(m_i, b_dA[:, None] * b_qj[None, :] * exp(b_k - b_zj[None, :]), 0.)
    p_dk = tl.make_block_ptr(dk + i_bh * T*K, (T, K), (K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
