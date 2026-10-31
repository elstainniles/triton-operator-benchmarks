# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: tests/ops/ascend_ops/cv_operator/chunk_gsa_fwd_k/op_impl.py:338 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
# Source archive: dependencies and launch adaptation are not executable here.
# The complete original file and dependency snapshots are under upstream/.
import triton
import triton.language as tl


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def chunk_gsa_fwd_k_kernel_intra(
    v,
    g,
    o,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    HQ: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BV: tl.constexpr,
    NC: tl.constexpr,
    NG: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // NG
    i_t, i_i = i_c // NC, i_c % NC
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    o_i = tl.arange(0, BC)
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V

    if i_t * BT + i_i * BC >= T:
        return

    p_g = tl.make_block_ptr(g + (bos * H + i_h) * V, (T, V), (H*V, 1), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
    p_gn = g + (bos + min(i_t * BT + i_i * BC, T)) * H*V + i_h * V + o_v
    b_gn = tl.load(p_gn, mask=m_v, other=0)
    b_o = tl.zeros([BC, BV], dtype=tl.float32)
    for i_j in range(0, i_i):
        p_A = tl.make_block_ptr(A + (bos*HQ+i_hq) * BT, (T, BT), (HQ*BT, 1), (i_t*BT+i_i*BC, i_j * BC), (BC, BC), (1, 0))
        p_v = tl.make_block_ptr(v + (bos*H+i_h) * V, (T, V), (H*V, 1), (i_t * BT + i_j * BC, i_v * BV), (BC, BV), (1, 0))
        p_gv = tl.make_block_ptr(g + (bos*H+i_h) * V, (T, V), (H*V, 1), (i_t * BT + i_j * BC, i_v * BV), (BC, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_gv = tl.load(p_gv, boundary_check=(0, 1))
        b_vg = (b_v * exp(b_gn[None, :] - b_gv)).to(b_v.dtype)
        b_A = tl.load(p_A, boundary_check=(0, 1))
        b_o += tl.dot(b_A, b_vg, allow_tf32=False)
    
    # if i_i > 0:
    #     for i_j in range(0, i_i):
    #         p_A = tl.make_block_ptr(A + (bos*HQ+i_hq) * BT, (T, BT), (HQ*BT, 1), (i_t*BT+i_i*BC, i_j * BC), (BC, BC), (1, 0))
    #         p_v = tl.make_block_ptr(v + (bos*H+i_h) * V, (T, V), (H*V, 1), (i_t * BT + i_j * BC, i_v * BV), (BC, BV), (1, 0))
    #         p_gv = tl.make_block_ptr(g + (bos*H+i_h) * V, (T, V), (H*V, 1), (i_t * BT + i_j * BC, i_v * BV), (BC, BV), (1, 0))
    #         b_v = tl.load(p_v, boundary_check=(0, 1))
    #         b_gv = tl.load(p_gv, boundary_check=(0, 1))
    #         b_vg = (b_v * exp(b_gn[None, :] - b_gv)).to(b_v.dtype)
    #         b_A = tl.load(p_A, boundary_check=(0, 1))
    #         b_o += tl.dot(b_A, b_vg, allow_tf32=False)
    b_g = tl.load(p_g, boundary_check=(0, 1))
    b_o *= exp(b_g - b_gn[None, :])

    o_A = (bos + i_t * BT + i_i * BC + tl.arange(0, BC)) * HQ*BT + i_hq * BT + i_i * BC
    m_A = (i_t * BT + i_i * BC + tl.arange(0, BC)) < T
    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        p_v = v + (bos + i_t * BT + i_i * BC + j) * H*V + i_h * V + o_v
        p_gv = g + (bos + i_t * BT + i_i * BC + j) * H*V + i_h * V + o_v
        b_A = tl.load(A + o_A + j, mask=m_A, other=0)
        b_v = tl.load(p_v, mask=m_v, other=0).to(tl.float32)
        b_gv = tl.load(p_gv, mask=m_v, other=0).to(tl.float32)
        b_vg = b_v[None, :] * exp(b_g - b_gv[None, :])
        b_o += tl.where(o_i[:, None] >= j, b_A[:, None] * b_vg, 0.)
    p_o = tl.make_block_ptr(o + (bos*HQ + i_hq) * V, (T, V), (HQ*V, 1), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
    b_o += tl.load(p_o, boundary_check=(0, 1))
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
