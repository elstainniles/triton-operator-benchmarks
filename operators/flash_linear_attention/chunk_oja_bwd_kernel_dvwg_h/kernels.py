# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: fla/ops/gated_oja_rule/chunk_h.py:664 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
# Source archive: dependencies and launch adaptation are not executable here.
# The complete original file and dependency snapshots are under upstream/.
import triton
import triton.language as tl


@triton.heuristics({
    'USE_GV': lambda args: args['gv'] is not None,
    'HAVE_GK': lambda args: args['dgk'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'K', 'V', 'BT', 'BK', 'BV', 'USE_GV'],
)
@triton.jit(do_not_specialize=['T'])
def chunk_oja_bwd_kernel_dvwg_h(
    k,
    v,
    gv,
    h,
    dh,
    dk,
    dw,
    dv,
    dgv_last,
    dgk,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_GV: tl.constexpr,
    HAVE_GK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    # offset calculation
    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    gv += (bos * H + i_h) * V
    h += (i_tg * H + i_h).to(tl.int64) * K*V
    dh += (i_tg * H + i_h).to(tl.int64) * K*V
    dk += (bos * H + i_h) * K
    dw += (bos * H + i_h) * V
    dv += (bos * H + i_h) * V
    dgv_last += (bos * H + i_h) * V

    b_dvg = tl.zeros([BT, BV], dtype=tl.float32)
    b_dw = tl.zeros([BT, BV], dtype=tl.float32)
    b_dgv_last = tl.zeros([BV,], dtype=tl.float32)

    if USE_GV:
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V
        p_gn = gv + (min(T, i_t * BT + BT) - 1) * H*V + o_v
        p_gv = tl.make_block_ptr(gv, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_gn = tl.load(p_gn, mask=m_v, other=0)
        b_gv = tl.load(p_gv, boundary_check=(0, 1))

    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dk = tl.make_block_ptr(dk, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_h = tl.make_block_ptr(h, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        p_dh = tl.make_block_ptr(dh, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))  # BT BK
        b_dk = tl.load(p_dk, boundary_check=(0, 1))  # BT BK
        b_h = tl.load(p_h, boundary_check=(0, 1))  # BK BV
        b_dh = tl.load(p_dh, boundary_check=(0, 1))  # BK BV

        b_dvg += tl.dot(b_k, b_dh.to(b_k.dtype))  # BT BK @ BK BV -> BT BV
        b_dw += tl.dot(b_dk.to(b_k.dtype), b_h.to(b_k.dtype))  # BT BK @ BK BV -> BT BV
        b_dgv_last += tl.sum((b_h * b_dh) * exp(b_gn), axis=0)

    if USE_GV:
        b_dv = b_dvg * exp(b_gn[None, :] - b_gv)

    p_v = tl.make_block_ptr(v, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_dv = tl.make_block_ptr(dv, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_dw = tl.make_block_ptr(dw, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_dgv_last = tl.make_block_ptr(dgv_last, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    b_v = tl.load(p_v, boundary_check=(0, 1))

    b_dgv_last += tl.sum(b_dv * b_v, axis=0)

    # 留给GSA2的接口
    if HAVE_GK:
        dgk += (bos * H + i_h) * V
        p_dgk = tl.make_block_ptr(dgk, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_dgk = tl.load(p_dgk, boundary_check=(0, 1))
        b_dgv_last = b_dgk + b_dgv_last[None, :]
    else:
        b_dgv_last = tl.zeros([BT, BV], dtype=tl.float32) + b_dgv_last[None, :]

    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dw, -b_dw.to(p_dw.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dgv_last, b_dgv_last.to(p_dgv_last.dtype.element_ty), boundary_check=(0, 1))
