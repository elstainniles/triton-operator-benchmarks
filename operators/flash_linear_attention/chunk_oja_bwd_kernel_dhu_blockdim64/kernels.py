# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: fla/ops/gated_oja_rule/chunk_h.py:265 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
# Source archive: dependencies and launch adaptation are not executable here.
# The complete original file and dependency snapshots are under upstream/.
import triton
import triton.language as tl


@triton.heuristics({
    'USE_GV': lambda args: args['gv'] is not None,
    'USE_INITIAL_STATE': lambda args: args['dh0'] is not None,
    'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [4, 3, 2]
        for BK in [64, 32]
    ],
    key=['H', 'K', 'V', 'BT', 'BK', 'USE_GV'],
    use_cuda_graph=use_cuda_graph,
)
@triton.jit(do_not_specialize=['T'])
def chunk_oja_bwd_kernel_dhu_blockdim64(
    q,
    vg,
    w,
    gv,
    dht,
    dh0,
    do,
    dh,
    dk,
    dk2,
    cu_seqlens,
    chunk_offsets,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    USE_GV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    IS_VARLEN: tl.constexpr
):
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
    b_dh1 = tl.zeros([BK, 64], dtype=tl.float32)
    if V > 64:
        b_dh2 = tl.zeros([BK, 64], dtype=tl.float32)
    if V > 128:
        b_dh3 = tl.zeros([BK, 64], dtype=tl.float32)
    if V > 192:
        b_dh4 = tl.zeros([BK, 64], dtype=tl.float32)

    # calculate offset
    q += ((bos * H + i_h) * K).to(tl.int64)
    vg += ((bos * H + i_h) * V).to(tl.int64)
    w += ((bos * H + i_h) * V).to(tl.int64)
    do += ((bos * H + i_h) * V).to(tl.int64)
    dk += ((bos * H + i_h) * K).to(tl.int64)
    dk2 += ((bos * H + i_h) * K).to(tl.int64)
    dh += ((boh * H + i_h) * K*V).to(tl.int64)
    if USE_GV:
        gv += ((bos * H + i_h) * V).to(tl.int64)

    stride_v = H*V
    stride_h = H*K*V
    stride_k = H*K
    if USE_INITIAL_STATE:
        dh0 += i_nh * K*V
    if USE_FINAL_STATE_GRADIENT:
        dht += i_nh * K*V

    if USE_FINAL_STATE_GRADIENT:
        p_dht1 = tl.make_block_ptr(dht, (K, V), (V, 1), (i_k * BK, 0), (BK, 64), (1, 0))  # [BK, BV]
        b_dh1 += tl.load(p_dht1, boundary_check=(0, 1))
        if V > 64:
            p_dht2 = tl.make_block_ptr(dht, (K, V), (V, 1), (i_k * BK, 64), (BK, 64), (1, 0))
            b_dh2 += tl.load(p_dht2, boundary_check=(0, 1))
        if V > 128:
            p_dht3 = tl.make_block_ptr(dht, (K, V), (V, 1), (i_k * BK, 128), (BK, 64), (1, 0))
            b_dh3 += tl.load(p_dht3, boundary_check=(0, 1))
        if V > 192:
            p_dht4 = tl.make_block_ptr(dht, (K, V), (V, 1), (i_k * BK, 192), (BK, 64), (1, 0))
            b_dh4 += tl.load(p_dht4, boundary_check=(0, 1))

    for i_t in range(NT - 1, -1, -1):
        p_dh1 = tl.make_block_ptr(dh + i_t*stride_h, (K, V), (V, 1), (i_k * BK, 0), (BK, 64), (1, 0))
        tl.store(p_dh1, b_dh1.to(p_dh1.dtype.element_ty), boundary_check=(0, 1))
        if V > 64:
            p_dh2 = tl.make_block_ptr(dh + i_t*stride_h, (K, V), (V, 1), (i_k * BK, 64), (BK, 64), (1, 0))
            tl.store(p_dh2, b_dh2.to(p_dh2.dtype.element_ty), boundary_check=(0, 1))
        if V > 128:
            p_dh3 = tl.make_block_ptr(dh + i_t*stride_h, (K, V), (V, 1), (i_k * BK, 128), (BK, 64), (1, 0))
            tl.store(p_dh3, b_dh3.to(p_dh3.dtype.element_ty), boundary_check=(0, 1))
        if V > 192:
            p_dh4 = tl.make_block_ptr(dh + i_t*stride_h, (K, V), (V, 1), (i_k * BK, 192), (BK, 64), (1, 0))
            tl.store(p_dh4, b_dh4.to(p_dh4.dtype.element_ty), boundary_check=(0, 1))

        last_idx = min((i_t + 1) * BT, T) - 1

        # Update dk_new, 按K切分
        p_dk = tl.make_block_ptr(dk, (T, K), (stride_k, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))  # [BT, BK]
        p_dk2 = tl.make_block_ptr(dk2, (T, K), (stride_k, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))  # [BT, BK]

        if V > 0:
            p_v = tl.make_block_ptr(vg, (T, V), (stride_v, 1), (i_t * BT, 0), (BT, 64), (1, 0))
            b_v = tl.load(p_v, boundary_check=(0, 1))  # [BT, BV]
            b_dk = tl.dot(b_v, tl.trans(b_dh1).to(b_v.dtype))  # [BT, BV] @ [BV, BK] -> [BT, BK]

        if V > 64:
            p_v = tl.make_block_ptr(vg, (T, V), (stride_v, 1), (i_t * BT, 64), (BT, 64), (1, 0))
            b_v = tl.load(p_v, boundary_check=(0, 1))
            b_dk += tl.dot(b_v, tl.trans(b_dh2).to(b_v.dtype))

        if V > 128:
            p_v = tl.make_block_ptr(vg, (T, V), (stride_v, 1), (i_t * BT, 128), (BT, 64), (1, 0))
            b_v = tl.load(p_v, boundary_check=(0, 1))
            b_dk += tl.dot(b_v, tl.trans(b_dh3).to(b_v.dtype))

        if V > 192:
            p_v = tl.make_block_ptr(vg, (T, V), (stride_v, 1), (i_t * BT, 192), (BT, 64), (1, 0))
            b_v = tl.load(p_v, boundary_check=(0, 1))
            b_dk += tl.dot(b_v, tl.trans(b_dh4).to(b_v.dtype))

        b_dk += tl.load(p_dk, boundary_check=(0, 1))

        tl.store(p_dk2, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

        # Update dh, 按照K切分，收集所有V维度，q一次就好，wdo要收集所有

        p_q = tl.make_block_ptr(q, (K, T), (1, stride_k), (i_k * BK, i_t * BT), (BK, BT), (0, 1))  # [BK, BT]
        b_q = tl.load(p_q, boundary_check=(0, 1))

        if V > 0:
            p_do = tl.make_block_ptr(do, (T, V), (stride_v, 1), (i_t * BT, 0), (BT, 64), (1, 0))  # [BT, BV]
            b_do = tl.load(p_do, boundary_check=(0, 1))
            p_w = tl.make_block_ptr(w, (T, V), (stride_v, 1), (i_t * BT, 0), (BT, 64), (1, 0))  # [BT, BV]
            b_w = tl.load(p_w, boundary_check=(0, 1))
            p_gv = tl.make_block_ptr(gv, (T, V), (stride_v, 1), (i_t * BT, 0), (BT, 64), (1, 0))  # [BT, BV]
            b_gv = tl.load(p_gv, boundary_check=(0, 1))
            if USE_GV:
                o_v1 = tl.arange(0, 64)
                b_gv_last1 = tl.load(gv + last_idx * H*V + o_v1, mask=(o_v1 < V), other=0.)
                b_dh1 *= exp(b_gv_last1[None, :])
                b_do *= exp(b_gv)
            b_dh1 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - \
                tl.dot(tl.trans(b_dk).to(b_w.dtype), b_w)  # [BK, BT] @ [BT, BV] - [BK, BT] @ [BT, BV]

        if V > 64:
            p_do = tl.make_block_ptr(do, (T, V), (stride_v, 1), (i_t * BT, 64), (BT, 64), (1, 0))
            b_do = tl.load(p_do, boundary_check=(0, 1))
            p_w = tl.make_block_ptr(w, (T, V), (stride_v, 1), (i_t * BT, 64), (BT, 64), (1, 0))  # [BT, BV]
            b_w = tl.load(p_w, boundary_check=(0, 1))
            p_gv = tl.make_block_ptr(gv, (T, V), (stride_v, 1), (i_t * BT, 64), (BT, 64), (1, 0))  # [BT, BV]
            b_gv = tl.load(p_gv, boundary_check=(0, 1))
            if USE_GV:
                o_v2 = 64 + o_v1
                b_gv_last2 = tl.load(gv + last_idx * H*V + o_v2, mask=(o_v2 < V), other=0.)
                b_dh2 *= exp(b_gv_last2[None, :])
                b_do *= exp(b_gv)
            b_dh2 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(tl.trans(b_dk).to(b_w.dtype), b_w)

        if V > 128:
            p_do = tl.make_block_ptr(do, (T, V), (stride_v, 1), (i_t * BT, 128), (BT, 64), (1, 0))
            b_do = tl.load(p_do, boundary_check=(0, 1))
            p_w = tl.make_block_ptr(w, (T, V), (stride_v, 1), (i_t * BT, 128), (BT, 64), (1, 0))  # [BT, BV]
            b_w = tl.load(p_w, boundary_check=(0, 1))
            p_gv = tl.make_block_ptr(gv, (T, V), (stride_v, 1), (i_t * BT, 128), (BT, 64), (1, 0))  # [BT, BV]
            b_gv = tl.load(p_gv, boundary_check=(0, 1))
            if USE_GV:
                o_v3 = 128 + o_v1
                b_gv_last3 = tl.load(gv + last_idx * H*V + o_v3, mask=(o_v3 < V), other=0.)
                b_dh3 *= exp(b_gv_last3[None, :])
                b_do *= exp(b_gv)
            b_dh3 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(tl.trans(b_dk).to(b_w.dtype), b_w)

        if V > 192:
            p_do = tl.make_block_ptr(do, (T, V), (stride_v, 1), (i_t * BT, 192), (BT, 64), (1, 0))
            b_do = tl.load(p_do, boundary_check=(0, 1))
            p_w = tl.make_block_ptr(w, (T, V), (stride_v, 1), (i_t * BT, 192), (BT, 64), (1, 0))  # [BT, BV]
            b_w = tl.load(p_w, boundary_check=(0, 1))
            p_gv = tl.make_block_ptr(gv, (T, V), (stride_v, 1), (i_t * BT, 192), (BT, 64), (1, 0))  # [BT, BV]
            b_gv = tl.load(p_gv, boundary_check=(0, 1))
            if USE_GV:
                o_v4 = 192 + o_v1
                b_gv_last4 = tl.load(gv + last_idx * H*V + o_v4, mask=(o_v4 < V), other=0.)
                b_dh4 *= exp(b_gv_last4[None, :])
                b_do *= exp(b_gv)
            b_dh4 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(tl.trans(b_dk).to(b_w.dtype), b_w)

    if USE_INITIAL_STATE:
        p_dh0 = tl.make_block_ptr(dh0, (K, V), (V, 1), (i_k * BK, 0), (BK, 64), (1, 0))
        tl.store(p_dh0, b_dh1.to(p_dh0.dtype.element_ty), boundary_check=(0, 1))
        if V > 64:
            p_dh1 = tl.make_block_ptr(dh0, (K, V), (V, 1), (i_k * BK, 64), (BK, 64), (1, 0))
            tl.store(p_dh1, b_dh2.to(p_dh1.dtype.element_ty), boundary_check=(0, 1))
        if V > 128:
            p_dh2 = tl.make_block_ptr(dh0, (K, V), (V, 1), (i_k * BK, 128), (BK, 64), (1, 0))
            tl.store(p_dh2, b_dh3.to(p_dh2.dtype.element_ty), boundary_check=(0, 1))
        if V > 192:
            p_dh3 = tl.make_block_ptr(dh0, (K, V), (V, 1), (i_k * BK, 192), (BK, 64), (1, 0))
            tl.store(p_dh3, b_dh4.to(p_dh3.dtype.element_ty), boundary_check=(0, 1))
