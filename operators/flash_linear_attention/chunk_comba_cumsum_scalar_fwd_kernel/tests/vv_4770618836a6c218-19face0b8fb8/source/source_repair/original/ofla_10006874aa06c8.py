# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import triton

import triton.language as tl

from ofla_74f55558e08fc1 import autotune_cache_kwargs

@triton.heuristics({
    'HAS_SCALE': lambda args: args['scale'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8]
    ],
    key=['B', 'H', 'BT', 'IS_VARLEN'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_comba_cumsum_scalar_fwd_kernel(
    g,
    g0,
    g1,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    p_g = tl.make_block_ptr(g + bos*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_g0 = tl.make_block_ptr(g0 + bos*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_g1 = tl.make_block_ptr(g1 + bos*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    # [BT]
    b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
    if HAS_SCALE:
        b_g = b_g * scale
    b_g1 = tl.cumsum(b_g, axis=0)
    b_g0 = b_g1 - b_g
    tl.store(p_g0, b_g0.to(p_g0.dtype.element_ty), boundary_check=(0,))
    tl.store(p_g1, b_g1.to(p_g1.dtype.element_ty), boundary_check=(0,))
