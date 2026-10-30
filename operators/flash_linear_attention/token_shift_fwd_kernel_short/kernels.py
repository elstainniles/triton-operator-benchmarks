# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: fla/modules/token_shift.py:68 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
# Source archive: dependencies and launch adaptation are not executable here.
# The complete original file and dependency snapshots are under upstream/.
import triton
import triton.language as tl


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    'USE_INITIAL_STATE': lambda args: args['cache'] is not None,
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
def token_shift_fwd_kernel_short(
    x,
    y,
    cu_seqlens,
    cache,
    cache_out,
    T,
    D: tl.constexpr,
    BD: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_DECODE: tl.constexpr,
):
    i_b, i_t = tl.program_id(0), tl.program_id(1)

    if IS_VARLEN:
        i_n = i_b
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        g_t = i_t + bos

        if g_t >= eos:
            return

        is_first_pos = (i_t == 0)
        is_last_pos = (g_t == eos - 1)
    else:
        g_t = i_t
        is_first_pos = (g_t == 0)
        is_last_pos = (g_t == T - 1)

    o_d = tl.arange(0, BD)
    m_d = o_d < D

    if IS_VARLEN:
        base_offset = g_t * D + o_d
    else:
        base_offset = i_b * T*D + g_t * D + o_d

    b_x = tl.load(x + base_offset, mask=m_d)
    if IS_VARLEN:
        cache_offset = i_n * D + o_d  # i_n is seq index
    else:
        cache_offset = i_b * D + o_d  # i_b is batch index

    if IS_DECODE and USE_INITIAL_STATE:
        b_cache = tl.load(cache + cache_offset, mask=m_d)
        delta = b_cache - b_x
        tl.store(y + base_offset, delta, mask=m_d)
        if STORE_FINAL_STATE:
            tl.store(cache_out + cache_offset, b_x, mask=m_d)
        return

    if is_first_pos:
        # First position in sequence: delta = -hidden_states
        if USE_INITIAL_STATE:
            # cache shape: [N, D]
            b_cache = tl.load(cache + cache_offset, mask=m_d)
            delta = b_cache - b_x
            tl.store(y + base_offset, delta, mask=m_d)
        else:
            tl.store(y + base_offset, -b_x, mask=m_d)
        return

    # Other positions: delta = prev - curr
    if IS_VARLEN:
        prev_offset = (g_t-1) * D + o_d
    else:
        prev_offset = i_b * T*D + (g_t-1) * D + o_d

    prev_values = tl.load(x + prev_offset, mask=m_d)
    delta = prev_values - b_x
    tl.store(y + base_offset, delta, mask=m_d)
    if STORE_FINAL_STATE:
        if is_last_pos:
            tl.store(cache_out + cache_offset, b_x, mask=m_d)
