# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: tests/ops/ascend_ops/cv_operator/parallel_rebased_fwd_kernel/test_parallel_rebased_fwd_kernel_npu.py:9 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
# Source archive: dependencies and launch adaptation are not executable here.
# The complete original file and dependency snapshots are under upstream/.
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['T'])
def parallel_rebased_fwd_kernel(
    q,
    k,
    v,
    o,
    z,
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
    # i_c: chunk index. used for sequence parallelism
    i_kv, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    NV = tl.cdiv(V, BV)
    i_k = i_kv // (NV)
    i_v = i_kv % (NV)

    p_q = tl.make_block_ptr(q + i_bh * T*K, (T, K), (K, 1), (i_c*BTL, i_k*BK), (BTL, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * T*K, (K, T), (1, K), (i_k*BK, 0), (BK, BTS), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * T*V, (T, V), (V, 1), (0, i_v*BV), (BTS, BV), (1, 0))

    # [BQ, BD] block Q, in the shared memory throughout the whole kernel
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)
    b_o = tl.zeros([BTL, BV], dtype=tl.float32)
    b_z = tl.zeros([BTL], dtype=tl.float32)

    # Q block and K block have no overlap
    # no need for mask, thereby saving flops
    for _ in range(0, i_c*BTL, BTS):
        # [BK, BTS]
        b_k = tl.load(p_k, boundary_check=(0, 1))

        # [BTS, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BTL, BTS]
        b_s = tl.dot(b_q, (b_k), allow_tf32=False)
        b_s = b_s * b_s
        b_z += tl.sum(b_s, axis=1)

        # [BQ, BD]
        b_o = b_o + tl.dot(b_s.to(b_v.dtype), b_v, allow_tf32=False)
        p_k = tl.advance(p_k, (0, BTS))
        p_v = tl.advance(p_v, (BTS, 0))

    # # rescale interchunk output
    tl.debug_barrier()
    o_q = tl.arange(0, BTL)
    # # sync threads, easy for compiler to optimize
    # tl.debug_barrier()

    o_k = tl.arange(0, BTS)
    p_k = tl.make_block_ptr(k + i_bh * T*K, (K, T), (1, K), (i_k*BK, i_c*BTL), (BK, BTS), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * T*V, (T, V), (V, 1), (i_c*BTL, i_v*BV), (BTS, BV), (1, 0))
    # Q block and K block have overlap. masks required
    for _ in range(i_c*BTL, (i_c + 1) * BTL, BTS):
        # [BK, BTS]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BTS, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BTL, BTS]
        m_s = o_q[:, None] >= o_k[None, :]
        b_s = tl.dot(b_q, b_k, allow_tf32=False)
        b_s = b_s * b_s
        b_s = tl.where(m_s, b_s, 0)
        b_z += tl.sum(b_s, axis=1)
        # [BTL, BV]
        b_o += tl.dot(b_s.to(b_q.dtype), b_v, allow_tf32=False)
        p_k = tl.advance(p_k, (0, BTS))
        p_v = tl.advance(p_v, (BTS, 0))
        o_k += BTS

    p_o = tl.make_block_ptr(o + (i_bh + B * H * i_k) * T*V, (T, V), (V, 1), (i_c*BTL, i_v*BV), (BTL, BV), (1, 0))
    p_z = z + (i_bh + B * H * i_k) * T + i_c*BTL + tl.arange(0, BTL)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_z, b_z.to(p_z.dtype.element_ty), mask=((i_c*BTL + tl.arange(0, BTL)) < T))
