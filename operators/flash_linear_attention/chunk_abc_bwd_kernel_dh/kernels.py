# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: tests/ops/ascend_ops/cv_operator/chunk_abc/chunk_abc.py:482 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
# Source archive: dependencies and launch adaptation are not executable here.
# The complete original file and dependency snapshots are under upstream/.
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['T'])
def chunk_abc_bwd_kernel_dh(
    q,
    z,
    do,
    dh,
    scale,
    T,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
    NORMK: tl.constexpr,
):
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    b_zp = tl.full([BK if NORMK else BV], float('inf'), dtype=tl.float32)
    for i_t in range(NT - 1, -1, -1):
        i_p = tl.maximum(i_t * BT - 1, 0)
        p_q = tl.make_block_ptr(q + i_bh * T*K, (K, T), (1, K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * T*V, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * NT*K*V + i_t * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))

        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)
        b_do = tl.load(p_do, boundary_check=(0, 1))

        tl.store(p_dh, b_dh.to(p_dh.dtype.element_ty), boundary_check=(0, 1))
        if NORMK:
            p_z = tl.make_block_ptr(z + i_bh * T*K, (K, T), (1, K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
            p_zc = tl.make_block_ptr(z + i_bh * T*K, (T * K,), (1,), (i_p * K + i_k * BK,), (BK,), (0,))
            b_zc = tl.load(p_zc, boundary_check=(0,))
            b_r, b_zp = exp(b_zc - b_zp), b_zc
            b_z = tl.load(p_z, boundary_check=(0, 1))
            b_q = (b_q * exp(b_zc[:, None] - b_z)).to(b_q.dtype)
            b_dh = b_dh * b_r[:, None]
        else:
            p_z = tl.make_block_ptr(z + i_bh * T*V, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_zc = tl.make_block_ptr(z + i_bh * T*V, (T * V,), (1,), (i_p * V + i_v * BV,), (BV,), (0,))
            b_zc = tl.load(p_zc, boundary_check=(0,))
            b_r, b_zp = exp(b_zc - b_zp), b_zc
            b_z = tl.load(p_z, boundary_check=(0,))
            b_do = (b_do * exp(b_zc[None, :] - b_z)).to(b_do.dtype)
            b_dh = b_dh * b_r[None, :]
        b_dh += tl.dot(b_q, b_do, allow_tf32=False)
