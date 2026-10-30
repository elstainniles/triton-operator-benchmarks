# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: fla/ops/attn/decoding.py:31 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
# Source archive: dependencies and launch adaptation are not executable here.
# The complete original file and dependency snapshots are under upstream/.
import triton
import triton.language as tl


@triton.heuristics({
    'USE_G': lambda args: args['g_cumsum'] is not None,
    'USE_SINK_BIAS': lambda args: args['sink_bias'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4] + ([] if check_shared_mem('hopper') else [8])
        for num_stages in [2, 3, 4, 5]
    ],
    key=['H', 'G', 'K', 'V', 'BK', 'BV', 'USE_G', 'USE_SINK_BIAS'],
    **autotune_cache_kwargs,
)
@triton.jit
def naive_attn_decoding_kernel(
    q,
    k,
    v,
    o,
    g_cumsum,
    sink_bias,
    scale,
    cu_seqlens,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_SINK_BIAS: tl.constexpr,
):
    i_v, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    bos, eos = tl.load(cu_seqlens + i_b).to(tl.int32), tl.load(cu_seqlens + i_b + 1).to(tl.int32)
    T = eos - bos

    p_q = tl.make_block_ptr(q + i_bh * K, (K,), (1, ), (0, ), (BK,), (0,))
    p_o = tl.make_block_ptr(o + i_bh * V, (V,), (1, ), (0, ), (BV,), (0,))

    b_q = tl.load(p_q, boundary_check=(0,))
    b_q = (b_q * scale).to(b_q.dtype)

    b_o = tl.zeros([BV], dtype=tl.float32)

    b_m = tl.full([1], float('-inf'), dtype=tl.float32)
    b_acc = tl.zeros([1], dtype=tl.float32)

    if USE_G:
        p_g = tl.make_block_ptr(g_cumsum + bos * HQ + i_hq, (T,), (HQ,), (T-1,), (1,), (0,))
        b_gq = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
    else:
        b_gq = None

    if USE_SINK_BIAS:
        b_sink_bias = tl.load(sink_bias + i_hq).to(tl.float32)
    else:
        b_sink_bias = None

    for i_s in range(0, T, BS):
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_s, 0), (BS, BK), (1, 0))
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H*V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
        # [BK, BS]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BS, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BT, BS]
        b_s = tl.sum(b_q[None, :] * b_k, 1)

        mask = i_s + tl.arange(0, BS) < T
        b_s = tl.where(mask, b_s, float('-inf'))

        if USE_G:
            p_gk = tl.make_block_ptr(g_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))
            b_gk = tl.load(p_gk, boundary_check=(0,)).to(tl.float32)
            b_s += b_gq - b_gk
        # [BT, BS]
        b_m, b_mp = tl.maximum(b_m, tl.max(b_s)), b_m
        b_r = exp(b_mp - b_m)
        # [BT, BS]
        b_p = exp(b_s - b_m)

        # [BT]
        b_acc = b_acc * b_r + tl.sum(b_p, 0)
        # [BT, BV]
        b_o = b_o * b_r + tl.sum(b_p[:, None] * b_v, 0)
        b_mp = b_m

    if USE_SINK_BIAS:
        # keep the sink-bias merge finite when masking leaves a row with no valid key.
        b_m = tl.where(b_m == float('-inf'), 0., b_m)
        b_acc += exp(b_sink_bias - b_m)
    b_o = b_o / b_acc
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, ))
