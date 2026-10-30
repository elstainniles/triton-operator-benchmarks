# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: fla/ops/ttt/chunk.py:138 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
# Source archive: dependencies and launch adaptation are not executable here.
# The complete original file and dependency snapshots are under upstream/.
import triton
import triton.language as tl


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3]
    ],
    key=['BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_ttt_linear_fwd_kernel_o(
    q,
    k,
    v,
    eta,
    h,
    hb,
    o,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
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
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    eta += bos * H + i_h
    o += (bos * H + i_h) * V
    h += (i_tg * H + i_h) * K * V
    hb += (i_tg * H + i_h) * V
    stride_qk = H*K
    stride_vo = H*V
    stride_eta = H

    p_q = tl.make_block_ptr(q, (T, K), (stride_qk, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k, (K, T), (1, stride_qk), (0, i_t * BT), (BK, BT), (0, 1))
    p_eta = tl.make_block_ptr(eta, (T,), (stride_eta,), (i_t * BT,), (BT,), (0,))
    p_h = tl.make_block_ptr(h, (K, V), (V, 1), (0, i_v * BV), (BK, BV), (1, 0))
    p_hb = tl.make_block_ptr(hb, (V,), (1,), (i_v * BV,), (BV,), (0,))
    # [BT, BK]
    b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
    # [BK, BT]
    b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
    # [BT, 1]
    b_eta = tl.load(p_eta, boundary_check=(0,), padding_option="zero")
    # [BK, BV]
    b_h = tl.load(p_h, boundary_check=(0, 1), padding_option="zero")
    # [BV]
    b_hb = tl.load(p_hb, boundary_check=(0,), padding_option="zero")
    # [BT, BK] @ [BK, BV] -> [BT, BV]
    b_o = tl.dot(b_q, b_h, allow_tf32=False)
    # [BT, BK] @ [BK, BT] -> [BT, BT]
    b_A = tl.dot(b_q, b_k, allow_tf32=False)

    o_i = tl.arange(0, BT)
    m_A = o_i[:, None] >= o_i[None, :]
    b_A = tl.where(m_A, b_A, 0)
    b_Ae = tl.where(m_A, b_eta[:, None], 0.0)

    p_v = tl.make_block_ptr(v, (T, V), (stride_vo, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o, (T, V), (stride_vo, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
    b_o = (b_o - tl.dot(b_eta[:, None] * b_A.to(b_v.dtype), b_v, allow_tf32=False)) * scale
    b_o += b_hb[None, :] - tl.dot(b_Ae.to(b_v.dtype), b_v, allow_tf32=False)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
