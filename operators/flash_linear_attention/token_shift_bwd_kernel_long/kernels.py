# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: fla/modules/token_shift.py:314 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
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
    key=['BD', 'NB'],
    **autotune_cache_kwargs,
)
@triton.jit
def token_shift_bwd_kernel_long(
    dx,
    dy,
    cu_seqlens,
    chunk_indices,
    grad_cache_in,
    grad_cache_out,
    T,
    D: tl.constexpr,
    BD: tl.constexpr,
    BT: tl.constexpr,
    NB: tl.constexpr,
    ND: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    HAS_DCACHE: tl.constexpr,
):
    i_dt, i_b = tl.program_id(0), tl.program_id(1)
    i_d, i_t_blk = i_dt % ND, i_dt // ND

    if IS_VARLEN:
        i_n, i_t_blk = tl.load(chunk_indices + i_t_blk * 2).to(tl.int32), \
            tl.load(chunk_indices + i_t_blk * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n), tl.load(cu_seqlens + i_n + 1)
        t_start = i_t_blk * BT
        t_end = tl.minimum(t_start + BT, eos - bos)
    else:
        i_n = i_b
        bos, eos = i_b * T, (i_b + 1) * T
        t_start = i_t_blk * BT
        t_end = tl.minimum(t_start + BT, T)

    o_d = i_d * BD + tl.arange(0, BD)
    m_d = o_d < D
    cache_off = i_n * D + o_d if IS_VARLEN else i_b * D + o_d

    for t in range(t_start, t_end):
        global_t = bos + t
        offset = global_t * D + o_d
        b_dy = tl.load(dy + offset, mask=m_d)

        if global_t == eos - 1:
            if HAS_DCACHE:
                b_dy_cache = tl.load(grad_cache_in + cache_off, mask=m_d)
                b_dx = -b_dy + b_dy_cache
            else:
                b_dx = -b_dy
        else:
            next_off = offset + D
            b_dx = -b_dy + tl.load(dy + next_off, mask=m_d)

        tl.store(dx + offset, b_dx, mask=m_d)

        if USE_INITIAL_STATE:
            if global_t == bos:
                tl.store(grad_cache_out + cache_off, b_dy, mask=m_d)
