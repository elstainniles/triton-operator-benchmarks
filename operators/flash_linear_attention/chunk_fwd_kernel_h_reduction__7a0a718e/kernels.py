# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: fla/ops/common/chunk_h_split.py:153 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
# Source archive: dependencies and launch adaptation are not executable here.
# The complete original file and dependency snapshots are under upstream/.
import triton
import triton.language as tl


@triton.heuristics({
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK, 'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64]
        for BV in [32, 64]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['BT', 'USE_G', 'USE_GK', 'USE_GV'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_fwd_kernel_h_reduction(
    g,
    gk,
    gv,
    hs,
    hr,
    ht,
    cu_seqlens,
    split_offsets,
    T,
    S: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GV: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NS = tl.cdiv(T, S)
        boh = tl.load(split_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NS = tl.cdiv(T, S)
        boh = i_n * NS

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    # skip the first split
    for i_s in range(1, NS):
        p_hs = tl.make_block_ptr(hs + ((boh + i_s-1) * H + i_h) * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        p_hr = tl.make_block_ptr(hr + ((boh + i_s) * H + i_h) * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h += tl.load(p_hs, boundary_check=(0, 1)).to(tl.float32)
        tl.store(p_hr, b_h.to(p_hr.dtype.element_ty), boundary_check=(0, 1))

        for i_t in range(tl.cdiv(i_s * S, BT), tl.cdiv(min(i_s * S + S, T), BT)):
            last_idx = min(i_t * BT + BT, T) - 1
            # scalar decay
            if USE_G:
                b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
                b_h *= exp(b_g_last)

            # vector decay, h = Diag(gk) @ h
            if USE_GK:
                p_gk_last = gk + (bos + last_idx) * H*K + i_h * K + i_k * BK + tl.arange(0, BK)
                b_gk_last = tl.load(p_gk_last, mask=(i_k * BK + tl.arange(0, BK) < K), other=0.)
                b_h *= exp(b_gk_last)[:, None]

            # vector decay, h = h @ Diag(gv)
            if USE_GV:
                p_gv_last = gv + (bos + last_idx) * H*V + i_h * V + i_v * BV + tl.arange(0, BV)
                b_gv_last = tl.load(p_gv_last, mask=(i_v * BV + tl.arange(0, BV) < V), other=0.)
                b_h *= exp(b_gv_last)[None, :]

    if NS > 1:
        if STORE_FINAL_STATE:
            p_hs = tl.make_block_ptr(hs + ((boh + NS-1) * H + i_h)*K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
            p_ht = tl.make_block_ptr(ht + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
            b_h += tl.load(p_hs, boundary_check=(0, 1)).to(tl.float32)
            tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
