# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: tests/ops/ascend_ops/cv_operator/fused_chunk_based_fwd_kernel/test_fused_chunk_based_fwd_kernel_npu.py:99 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
# Source archive: dependencies and launch adaptation are not executable here.
# The complete original file and dependency snapshots are under upstream/.
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['T'])
def fused_chunk_based_fwd_kernel(
    q,
    k,
    v,
    o,
    z,
    scale,  # K ** -0.5
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    o_i = tl.arange(0, BT)

    # [BT, BT]
    m_s = o_i[:, None] >= o_i[None, :]

    # [BV], zero-order taylor expansion
    b_h_0o = tl.zeros([BV], dtype=tl.float32)
    # [BK, BV], first-order taylor expansion
    b_h_1o = tl.zeros([BK, BV], dtype=tl.float32)
    # [BK, BK, BV] second-order taylor expansion
    b_h_2o = tl.zeros([BK*BK, BV], dtype=tl.float32)

    # make block pointers
    p_q = tl.make_block_ptr(q + i_bh * T*K, (T, K), (K, 1), (0, i_k * BK), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * T*K, (K, T), (1, K), (i_k * BK, 0), (BK, BT), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * T*V, (T, V), (V, 1), (0, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o + (i_bh + i_k*B*H) * T*V, (T, V), (V, 1), (0, i_v * BV), (BT, BV), (1, 0))

    p_z = z + (i_bh + i_k * B * H) * T + tl.arange(0, BT)
    k_2o = tl.zeros([1, BK * BK], dtype=tl.float32)
    k_1o = tl.zeros([1, BK], dtype=tl.float32)
    k_0o = 0

    for i in range(0, tl.cdiv(T, BT)):
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BK*BK, BT]
        b_k_2o = b_k[:, None, :] * b_k[None, :, :]
        b_k_2o = tl.reshape(b_k_2o, [BK * BK, BT]).to(b_k.dtype)
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BT, BK]
        b_q = (tl.load(p_q, boundary_check=(0, 1)) * scale).to(b_k.dtype)
        b_o = tl.zeros([BT, BV], dtype=tl.float32)
        b_z = tl.zeros([BT], dtype=tl.float32)

        # interchunk
        # zero-order
        b_o += b_h_0o
        b_z += k_0o
        # first-order
        b_o += tl.dot(b_q, b_h_1o.to(b_q.dtype), allow_tf32=False)
        b_z += tl.sum(b_q * k_1o, axis=1)
        # second-order
        b_q_2o = b_q[:, :, None] * b_q[:, None, :]
        b_q_2o = tl.reshape(b_q_2o, [BT, BK * BK]).to(b_k.dtype)
        b_o += tl.dot(b_q_2o, b_h_2o.to(b_q_2o.dtype), allow_tf32=False) * 0.5
        b_z += tl.sum(b_q_2o * k_2o, axis=1) * 0.5

        # update running statistics
        k_1o += tl.sum(b_k, axis=1)[None, :]
        k_2o += tl.sum(b_k_2o, axis=1)[None, :]
        k_0o += BT

        # intrachunk
        # [BT, BT]
        b_s = tl.dot(b_q, b_k, allow_tf32=False)
        b_s = 1 + b_s + 0.5 * b_s * b_s
        b_s = tl.where(m_s, b_s, 0)
        b_z += tl.sum(b_s, axis=1)
        b_o += tl.dot(b_s.to(b_q.dtype), b_v, allow_tf32=False)
        # [TB, BV]
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_z, b_z.to(p_z.dtype.element_ty), mask=(i * BT + tl.arange(0, BT)) < T)

        # update hidden state
        # [BK, BV]
        b_h_2o = b_h_2o + tl.dot(b_k_2o.to(b_v.dtype), b_v, allow_tf32=False)
        b_h_1o = b_h_1o + tl.dot(b_k, b_v, allow_tf32=False)
        b_h_0o = b_h_0o + tl.sum(b_v, axis=0)

        p_q = tl.advance(p_q, (BT, 0))
        p_k = tl.advance(p_k, (0, BT))
        p_v = tl.advance(p_v, (BT, 0))
        p_o = tl.advance(p_o, (BT, 0))
        p_z += BT
