# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: tests/ops/ascend_ops/cv_operator/chunk_transform_qk_fwd_kernel/test_chunk_transform_qk_fwd_kernel_npu.py:75 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
# Source archive: dependencies and launch adaptation are not executable here.
# The complete original file and dependency snapshots are under upstream/.
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4]
    ],
    key=['BT', 'K', 'V'],
)
@triton.jit(do_not_specialize=['T'])
def chunk_transform_qk_fwd_kernel(
    q,
    k,
    v,
    beta,
    o,
    A,
    q_new,
    k_new,
    A_local,
    scale,
    T,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BT: tl.constexpr,
    OUTPUT_ATTENTIONS: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)

    p_q = tl.make_block_ptr(q + i_bh * T*K, (T, K), (K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * T*K, (T, K), (K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * T*V, (T, V), (V, 1), (i_t * BT, 0), (BT, BV), (1, 0))
    b_q = (tl.load(p_q, boundary_check=(0, 1)) * scale).to(p_q.dtype.element_ty)
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_v = tl.load(p_v, boundary_check=(0, 1))

    p_T = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_T = tl.load(p_T, boundary_check=(0, 1))

    o_i = tl.arange(0, BT)
    m_t = o_i[:, None] >= o_i[None, :]
    b_qk = tl.where(m_t, tl.dot(b_q, tl.trans(b_k), allow_tf32=False), 0).to(b_q.dtype)
    m_t = o_i[:, None] > o_i[None, :]
    b_kk = tl.where(m_t, tl.dot(b_k, tl.trans(b_k), allow_tf32=False), 0).to(b_k.dtype)

    p_beta = tl.make_block_ptr(beta + i_bh * T, (T, ), (1, ), (i_t * BT, ), (BT, ), (0, ))
    b_beta = tl.load(p_beta, boundary_check=(0, ))
    b_k_beta = (b_k * b_beta[:, None]).to(b_k.dtype)

    b_qkT = tl.dot(b_qk, b_T, allow_tf32=False).to(b_k.dtype)

    if OUTPUT_ATTENTIONS:
        p_a = tl.make_block_ptr(A_local + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
        tl.store(p_a, b_qkT.to(p_a.dtype.element_ty), boundary_check=(0, 1))

    b_kkT = tl.dot(b_kk, b_T, allow_tf32=False).to(b_k.dtype)
    p_o = tl.make_block_ptr(o + i_bh * T*V, (T, V), (V, 1), (i_t * BT, 0), (BT, BV), (1, 0))
    tl.store(p_o, tl.dot(b_qkT, b_v).to(p_o.dtype.element_ty), boundary_check=(0, 1))

    p_q_new = tl.make_block_ptr(q_new + i_bh * T*K, (T, K), (K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    tl.store(p_q_new, (b_q - tl.dot(b_qkT, b_k_beta, allow_tf32=False)).to(p_q_new.dtype.element_ty), boundary_check=(0, 1))

    p_k_new = tl.make_block_ptr(k_new + i_bh * T*K, (T, K), (K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    b_k_new = b_k - tl.dot(tl.trans(b_kkT), b_k_beta, allow_tf32=False)
    tl.store(p_k_new, b_k_new.to(p_k_new.dtype.element_ty), boundary_check=(0, 1))
