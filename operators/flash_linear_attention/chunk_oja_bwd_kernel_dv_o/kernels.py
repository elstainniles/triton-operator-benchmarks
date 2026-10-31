# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: fla/ops/gated_oja_rule/chunk_o.py:543 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
# Source archive: dependencies and launch adaptation are not executable here.
# The complete original file and dependency snapshots are under upstream/.
import triton
import triton.language as tl


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.jit(do_not_specialize=['T'])
def chunk_oja_bwd_kernel_dv_o(
    v,
    g,
    o,
    A,
    do,
    dv,
    dv2,
    dg,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BV: tl.constexpr,
    NC: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    i_t, i_i = i_c // NC, i_c % NC
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V

    if i_t * BT + i_i * BC >= T:
        return

    p_gv = tl.make_block_ptr(g + (bos*H+i_h)*V, (T, V), (H*V, 1), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
    p_gn = g + (bos + min(i_t * BT + i_i * BC + BC, T)-1)*H*V + i_h*V + o_v
    # [BV,]
    b_gn = tl.load(p_gn, mask=m_v, other=0)
    # [BC, BV]
    b_gv = tl.load(p_gv, boundary_check=(0, 1))
    b_dvg = tl.zeros([BC, BV], dtype=tl.float32)
    for i_j in range(i_i + 1, NC):
        p_g = tl.make_block_ptr(g + (bos*H+i_h) * V, (T, V), (H*V, 1), (i_t * BT + i_j * BC, i_v * BV), (BC, BV), (1, 0))
        p_A = tl.make_block_ptr(A + (bos*H+i_h) * BT, (BT, T), (1, H*BT), (i_i*BC, i_t*BT + i_j*BC), (BC, BC), (0, 1))
        p_do = tl.make_block_ptr(do + (bos*H+i_h) * V, (T, V), (H*V, 1), (i_t*BT + i_j*BC, i_v*BV), (BC, BV), (1, 0))
        # [BC, BV]
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1)) * exp(b_g - b_gn[None, :])
        # [BC, BC]
        b_A = tl.load(p_A, boundary_check=(0, 1))
        # [BC, BV]
        b_dvg += tl.dot(b_A, b_do.to(b_A.dtype))
    b_dv = b_dvg * exp(b_gn[None, :] - b_gv)

    o_i = tl.arange(0, BC)
    o_c = i_i * BC + tl.arange(0, BC)

    p_g = g + (bos + i_t * BT + i_i * BC) * H*V + i_h * V + o_v
    p_A = A + (bos + i_t*BT + i_i*BC) * H*BT + i_h * BT + o_c
    p_do = do + (bos + i_t*BT + i_i*BC) * H*V + i_h * V + o_v
    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        # [BC,]
        b_A = tl.load(p_A)
        # [BV,]
        b_g = tl.load(p_g, mask=m_v, other=0)
        b_do = tl.load(p_do, mask=m_v, other=0)
        # [BC, BV]
        m_i = o_i[:, None] <= j
        b_dv += tl.where(m_i, exp(b_g[None, :] - b_gv) * b_A[:, None] * b_do[None, :], 0.)

        p_g += H * V
        p_A += H * BT
        p_do += H * V
    p_o = tl.make_block_ptr(o + (bos*H+i_h)*V, (T, V), (H*V, 1), (i_t*BT + i_i*BC, i_v*BV), (BC, BV), (1, 0))
    p_v = tl.make_block_ptr(v + (bos*H+i_h)*V, (T, V), (H*V, 1), (i_t*BT + i_i*BC, i_v*BV), (BC, BV), (1, 0))
    p_do = tl.make_block_ptr(do + (bos*H+i_h)*V, (T, V), (H*V, 1), (i_t*BT + i_i*BC, i_v*BV), (BC, BV), (1, 0))
    p_dv = tl.make_block_ptr(dv + (bos*H+i_h)*V, (T, V), (H*V, 1), (i_t*BT + i_i*BC, i_v*BV), (BC, BV), (1, 0))
    p_dv2 = tl.make_block_ptr(dv2 + (bos*H+i_h)*V, (T, V), (H*V, 1), (i_t*BT + i_i*BC, i_v*BV), (BC, BV), (1, 0))
    p_dg = tl.make_block_ptr(dg + (bos*H+i_h)*V, (T, V), (H*V, 1), (i_t*BT + i_i*BC, i_v*BV), (BC, BV), (1, 0))

    b_o = tl.load(p_o, boundary_check=(0, 1)).to(tl.float32)
    b_v = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32)
    b_do = tl.load(p_do, boundary_check=(0, 1)).to(tl.float32)
    b_dv = b_dv + tl.load(p_dv, boundary_check=(0, 1)).to(tl.float32)
    b_dg = b_o * b_do - b_v * b_dv
    tl.store(p_dv2, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0, 1))
