# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: fla/modules/token_shift.py:236 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
# Source archive: dependencies and launch adaptation are not executable here.
# The complete original file and dependency snapshots are under upstream/.
import triton
import triton.language as tl


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    'USE_INITIAL_STATE': lambda args: args['grad_cache_out'] is not None,
    'HAS_DCACHE': lambda args: args['grad_cache_in'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS_AUTOTUNE
        for num_stages in [1, 2, 3]
    ],
    key=['BD'],
    **autotune_cache_kwargs,
)
@triton.jit
def token_shift_bwd_kernel_short(
    dx,
    dy,
    cu_seqlens,
    grad_cache_in,
    grad_cache_out,
    T,
    D: tl.constexpr,
    BD: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    HAS_DCACHE: tl.constexpr,
):
    i_b, i_t = tl.program_id(0), tl.program_id(1)

    if IS_VARLEN:
        i_n = i_b
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        g_t = i_t + bos
        if g_t >= eos:
            return
        is_first_pos = (g_t == bos)
        is_last_pos = (g_t == eos - 1)
    else:
        g_t = i_t
        is_first_pos = (g_t == 0)
        is_last_pos = (g_t == T - 1)

    o_d = tl.arange(0, BD)
    m_d = o_d < D

    if IS_VARLEN:
        base_offset = g_t * D + o_d
        # This should not be used for varlen
        cache_off = i_n * D + o_d
    else:
        base_offset = i_b * T * D + g_t * D + o_d
        cache_off = i_b * D + o_d

    b_dy = tl.load(dy + base_offset, mask=m_d)

    if is_last_pos:
        # grad = -grad_delta[t] + grad_cache_in（from next rank）
        if HAS_DCACHE:
            b_dy_cache = tl.load(grad_cache_in + cache_off, mask=m_d)
            b_dx = -b_dy + b_dy_cache
        else:
            b_dx = -b_dy
    else:
        # grad = -grad_delta[t] + grad_delta[t+1]
        if IS_VARLEN:
            next_offset = (g_t + 1) * D + o_d
        else:
            next_offset = i_b * T * D + (g_t + 1) * D + o_d
        b_dx = -b_dy + tl.load(dy + next_offset, mask=m_d)

    tl.store(dx + base_offset, b_dx, mask=m_d)

    if USE_INITIAL_STATE:
        if is_first_pos:
            tl.store(grad_cache_out + cache_off, b_dy, mask=m_d)
