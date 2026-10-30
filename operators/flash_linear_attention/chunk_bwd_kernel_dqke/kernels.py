# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: fla/ops/ttt/chunk.py:566 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
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
    key=['BT', 'BK', 'BV'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_bwd_kernel_dqke(
    q,
    k,
    v,
    e,
    h,
    do,
    dh,
    dhb,
    dq,
    dk,
    de,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
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
    v += (bos * H + i_h) * V
    do += (bos * H + i_h) * V
    h += (i_tg * H + i_h) * K * V
    dh += (i_tg * H + i_h) * K * V
    dhb += (i_tg * H + i_h) * V
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    dq += (bos * H + i_h) * K
    dk += (bos * H + i_h) * K
    e += bos * H + i_h
    de += bos * H + i_h
    stride_qk = H*K
    stride_vo = H*V
    stride_e = H

    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_ds = tl.zeros([BT, BT], dtype=tl.float32)
    b_de = tl.zeros([BT], dtype=tl.float32)

    p_k = tl.make_block_ptr(k, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    p_e_last = (e + (i_t*BT+BT-1)*stride_e) if (i_t*BT+BT) <= T else (e + (T-1)*stride_e)
    i_last = (BT-1) if (i_t*BT+BT) <= T else (T % BT-1)
    mask = (tl.arange(0, BT) == i_last)
    b_e_last = tl.load(p_e_last)

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v, (T, V), (stride_vo, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_do = tl.make_block_ptr(do, (T, V), (stride_vo, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        p_dh = tl.make_block_ptr(dh, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        p_dhb = tl.make_block_ptr(dhb, (V,), (1,), (i_v * BV,), (BV,), (0,))
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        # [BV, BK]
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_dh = tl.load(p_dh, boundary_check=(0, 1))
        # [BV]
        b_dhb = tl.load(p_dhb, boundary_check=(0,))
        # [BT, BV] @ [BV, BT] -> [BT, BT]
        b_ds += tl.dot(b_do, tl.trans(b_v))
        # [BT, BV] @ [BV, BK] -> [BT, BK]
        b_dq += tl.dot(b_do, b_h.to(b_do.dtype))
        # [BT, BV] @ [BV, BK] -> [BT, BK]
        b_dk -= b_e_last * tl.dot(b_v, b_dh.to(b_v.dtype))
        b_de -= mask * tl.sum(tl.trans(b_dh) * tl.dot(tl.trans(b_k), b_v.to(b_k.dtype)))
        b_de -= mask * tl.sum(b_dhb * tl.sum(b_v, axis=0).to(b_k.dtype))

    o_i = tl.arange(0, BT)
    p_q = tl.make_block_ptr(q, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_e = tl.make_block_ptr(e, (T,), (stride_e,), (i_t * BT,), (BT,), (0,))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_e = tl.load(p_e, boundary_check=(0,))

    p_dq = tl.make_block_ptr(dq, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_de = tl.make_block_ptr(de, (T,), (stride_e,), (i_t * BT,), (BT,), (0,))

    b_ds = tl.where(o_i[:, None] >= o_i[None, :], b_ds, 0)
    b_ds = b_ds.to(b_k.dtype)
    b_dq -= tl.dot(b_ds, b_k) * b_e[:, None]
    b_dk -= tl.dot(tl.trans(b_ds), b_q * b_e[:, None]) * scale
    b_de -= tl.sum(scale * tl.dot(b_ds, b_k) * b_q, axis=1)
    b_de -= tl.sum(b_ds, axis=1)
    b_dq *= scale
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_de, b_de.to(p_de.dtype.element_ty), boundary_check=(0,))
