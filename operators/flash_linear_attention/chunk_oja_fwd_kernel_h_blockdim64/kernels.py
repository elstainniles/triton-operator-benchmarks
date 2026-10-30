# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: fla/ops/gated_oja_rule/chunk_h.py:38 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
# Source archive: dependencies and launch adaptation are not executable here.
# The complete original file and dependency snapshots are under upstream/.
import triton
import triton.language as tl


@triton.heuristics({
    'USE_GV': lambda args: args['gv'] is not None,
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'SAVE_NEW_KEY': lambda args: args['k_new'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [2, 3, 4]
        for BK in [32, 64]
    ],
    key=['H', 'K', 'V', 'BT'],
    use_cuda_graph=use_cuda_graph,
)
@triton.jit(do_not_specialize=['T'])
def chunk_oja_fwd_kernel_h_blockdim64(
    v,
    u,
    w,
    k_new,
    gv,
    h,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    USE_GV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_KEY: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    # (triton.cdiv(K, meta['BK']), N*H)
    i_k, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    # [BK, BV]
    b_h1 = tl.zeros([BK, 64], dtype=tl.float32)
    if V > 64:
        b_h2 = tl.zeros([BK, 64], dtype=tl.float32)
    if V > 128:
        b_h3 = tl.zeros([BK, 64], dtype=tl.float32)
    if V > 192:
        b_h4 = tl.zeros([BK, 64], dtype=tl.float32)

    # calculate offset
    h += ((boh * H + i_h) * K*V).to(tl.int64)
    v += ((bos * H + i_h) * V).to(tl.int64)
    u += ((bos * H + i_h) * K).to(tl.int64)
    w += ((bos * H + i_h) * V).to(tl.int64)
    if SAVE_NEW_KEY:
        k_new += ((bos * H + i_h) * K).to(tl.int64)
    stride_v = H*V
    stride_h = H*K*V
    stride_k = H*K
    if USE_INITIAL_STATE:
        h0 = h0 + i_nh * K*V
    if STORE_FINAL_STATE:
        ht = ht + i_nh * K*V

    # load initial state
    if USE_INITIAL_STATE:
        p_h0_1 = tl.make_block_ptr(h0, (K, V), (V, 1), (i_k * BK, 0), (BK, 64), (1, 0))
        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
        if V > 64:
            p_h0_2 = tl.make_block_ptr(h0, (K, V), (V, 1), (i_k * BK, 64), (BK, 64), (1, 0))
            b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
        if V > 128:
            p_h0_3 = tl.make_block_ptr(h0, (K, V), (V, 1), (i_k * BK, 128), (BK, 64), (1, 0))
            b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
        if V > 192:
            p_h0_4 = tl.make_block_ptr(h0, (K, V), (V, 1), (i_k * BK, 192), (BK, 64), (1, 0))
            b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)

    # main recurrence
    for i_t in range(NT):
        p_h1 = tl.make_block_ptr(h + i_t * stride_h, (K, V), (V, 1), (i_k * BK, 0), (BK, 64), (1, 0))
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
        if V > 64:
            p_h2 = tl.make_block_ptr(h + i_t * stride_h, (K, V), (V, 1), (i_k * BK, 64), (BK, 64), (1, 0))
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))
        if V > 128:
            p_h3 = tl.make_block_ptr(h + i_t * stride_h, (K, V), (V, 1), (i_k * BK, 128), (BK, 64), (1, 0))
            tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(0, 1))
        if V > 192:
            p_h4 = tl.make_block_ptr(h + i_t * stride_h, (K, V), (V, 1), (i_k * BK, 192), (BK, 64), (1, 0))
            tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(0, 1))

        p_w = tl.make_block_ptr(w, (T, V), (stride_v, 1), (i_t * BT, 0), (BT, 64), (1, 0))
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_k = tl.dot(b_w, tl.trans(b_h1).to(b_w.dtype))  # BT BK
        if V > 64:
            p_w = tl.make_block_ptr(w, (T, V), (stride_v, 1), (i_t * BT, 64), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_k += tl.dot(b_w, tl.trans(b_h2).to(b_w.dtype))
        if V > 128:
            p_w = tl.make_block_ptr(w, (T, V), (stride_v, 1), (i_t * BT, 128), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_k += tl.dot(b_w, tl.trans(b_h3).to(b_w.dtype))
        if V > 192:
            p_w = tl.make_block_ptr(w, (T, V), (stride_v, 1), (i_t * BT, 192), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_k += tl.dot(b_w, tl.trans(b_h4).to(b_w.dtype))

        p_u = tl.make_block_ptr(u, (T, K), (stride_k, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_u, boundary_check=(0, 1)) - b_k

        if SAVE_NEW_KEY:
            p_k = tl.make_block_ptr(k_new, (T, K), (stride_k, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            tl.store(p_k, b_k.to(p_k.dtype.element_ty), boundary_check=(0, 1))

        last_idx = min((i_t + 1) * BT, T) - 1

        if USE_GV:
            o_v1 = tl.arange(0, 64)
            b_gk_last1 = tl.load(gv + (bos + last_idx) * H*V + i_h * V + o_v1, mask=(o_v1 < V), other=0.)
            b_h1 *= exp(b_gk_last1)[None, :]
            if V > 64:
                o_v2 = 64 + o_v1
                b_gk_last2 = tl.load(gv + (bos + last_idx) * H*V + i_h * V + o_v2, mask=(o_v2 < V), other=0.)
                b_h2 *= exp(b_gk_last2)[None, :]
            if V > 128:
                o_v3 = 128 + o_v1
                b_gk_last3 = tl.load(gv + (bos + last_idx) * H*V + i_h * V + o_v3, mask=(o_v3 < V), other=0.)
                b_h3 *= exp(b_gk_last3)[None, :]
            if V > 192:
                o_v4 = 192 + o_v1
                b_gk_last4 = tl.load(gv + (bos + last_idx) * H*V + i_h * V + o_v4, mask=(o_v4 < V), other=0.)
                b_h4 *= exp(b_gk_last4)[None, :]

        b_k = b_k.to(v.dtype.element_ty)  # BT BK

        p_v = tl.make_block_ptr(v, (T, V), (stride_v, 1), (i_t * BT, 0), (BT, 64), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))  # BT BV
        b_h1 += tl.dot(tl.trans(b_k), b_v)
        if V > 64:
            p_v = tl.make_block_ptr(v, (T, V), (stride_v, 1), (i_t * BT, 64), (BT, 64), (1, 0))
            b_v = tl.load(p_v, boundary_check=(0, 1))
            b_h2 += tl.dot(tl.trans(b_k), b_v)
        if V > 128:
            p_v = tl.make_block_ptr(v, (T, V), (stride_v, 1), (i_t * BT, 128), (BT, 64), (1, 0))
            b_v = tl.load(p_v, boundary_check=(0, 1))
            b_h3 += tl.dot(tl.trans(b_k), b_v)
        if V > 192:
            p_v = tl.make_block_ptr(v, (T, V), (stride_v, 1), (i_t * BT, 192), (BT, 64), (1, 0))
            b_v = tl.load(p_v, boundary_check=(0, 1))
            b_h4 += tl.dot(tl.trans(b_k), b_v)
    # epilogue
    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (i_k * BK, 0), (BK, 64), (1, 0))
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if V > 64:
            p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (i_k * BK, 64), (BK, 64), (1, 0))
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if V > 128:
            p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (i_k * BK, 128), (BK, 64), (1, 0))
            tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if V > 192:
            p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (i_k * BK, 192), (BK, 64), (1, 0))
            tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
