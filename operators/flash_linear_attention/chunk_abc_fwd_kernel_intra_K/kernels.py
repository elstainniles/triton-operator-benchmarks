# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: tests/ops/ascend_ops/cv_operator/chunk_abc/chunk_abc.py:291 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
# Source archive: dependencies and launch adaptation are not executable here.
# The complete original file and dependency snapshots are under upstream/.
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['T'])
def chunk_abc_fwd_kernel_intra_K(
    v,
    z,
    o,
    A,
    T,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BV: tl.constexpr,
    NC: tl.constexpr,
):
    i_v, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_t, i_i = i_c // NC, i_c % NC

    p_z = tl.make_block_ptr(z + i_bh * T*V, (T, V), (V, 1), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
    p_zn = tl.make_block_ptr(z + i_bh * T*V, (T * V,), (1,), ((i_t * BT + i_i * BC) * V + i_v * BV,), (BV,), (0,))
    b_zn = tl.load(p_zn, boundary_check=(0,))
    b_o = tl.zeros([BC, BV], dtype=tl.float32)
    for i_j in range(0, i_i):
        p_A = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
        p_v = tl.make_block_ptr(v + i_bh * T*V, (T, V), (V, 1), (i_t * BT + i_j * BC, i_v * BV), (BC, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_A = tl.load(p_A, boundary_check=(0, 1))
        b_o += tl.dot(b_A, exp(b_v - b_zn[None, :]).to(b_v.dtype), allow_tf32=False)
    b_z = tl.load(p_z, boundary_check=(0, 1))
    b_o *= exp(b_zn[None, :] - b_z)

    o_i = tl.arange(0, BC)
    o_A = i_bh * T * BT + (i_t * BT + i_i * BC + tl.arange(0, BC)) * BT + i_i * BC
    m_A = (i_t * BT + i_i * BC + tl.arange(0, BC)) < T
    for j in range(0, BC):
        p_v = tl.make_block_ptr(v + i_bh * T*V, (T * V,), (1,), ((i_t * BT + i_i * BC + j) * V + i_v * BV,), (BV,), (0,))
        b_A = tl.load(A + o_A + j, mask=m_A, other=0)
        b_v = tl.load(p_v, boundary_check=(0,)).to(tl.float32)
        m_i = o_i[:, None] >= j
        b_o += tl.where(m_i, b_A[:, None] * exp(b_v[None, :] - b_z), 0)
    p_o = tl.make_block_ptr(o + i_bh * T*V, (T, V), (V, 1), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
