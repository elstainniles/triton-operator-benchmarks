# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: tests/ops/ascend_ops/vv_operator/fused_recurrent_hgrn_bwd_kernel/test_fused_recurrent_hgrn_bwd_kernel_npu.py:215 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
# Source archive: dependencies and launch adaptation are not executable here.
# The complete original file and dependency snapshots are under upstream/.
import triton
import triton.language as tl


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BD': BD}, num_warps=num_warps)
        for BD in [32, 64, 128]
        for num_warps in [1, 2, 4, 8]
    ],
    key=['D'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_hgrn_bwd_kernel(
    g,
    o,
    h0,
    dx,
    dg,
    do,
    dht,
    dh0,
    cu_seqlens,
    T,
    D: tl.constexpr,
    BD: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_d, i_n = tl.program_id(0), tl.program_id(1)
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T

    o_d = i_d * BD + tl.arange(0, BD)
    mask = o_d < D

    p_g = g + (bos + T - 1) * D + o_d
    p_o = o + (bos + T - 2) * D + o_d
    p_dx = dx + (bos + T - 1) * D + o_d
    p_dg = dg + (bos + T - 1) * D + o_d
    p_do = do + (bos + T - 1) * D + o_d

    b_dh = tl.zeros([BD], dtype=tl.float32)
    if USE_FINAL_STATE_GRADIENT:
        p_dht = dht + i_n * D + o_d
        b_dh += tl.load(p_dht, mask=mask, other=0).to(tl.float32)

    for i in range(T - 1, -1, -1):
        b_g = tl.load(p_g, mask=mask, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask, other=0).to(tl.float32)
        if i > 0:
            b_o = tl.load(p_o, mask=mask, other=0).to(tl.float32)
        elif USE_INITIAL_STATE:
            b_o = tl.load(h0 + i_n * D + o_d, mask=mask, other=0).to(tl.float32)
        else:
            b_o = tl.zeros([BD], dtype=tl.float32)

        b_dh = b_dh + b_do
        b_dx = b_dh
        b_dh = b_dh * exp(b_g)
        b_dg = b_dh * b_o
        tl.store(p_dx, b_dx.to(p_dx.dtype.element_ty), mask=mask)
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), mask=mask)

        p_g -= D
        p_o -= D
        p_dx -= D
        p_dg -= D
        p_do -= D

    if USE_INITIAL_STATE:
        p_dh0 = dh0 + i_n * D + o_d
        tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), mask=mask)
