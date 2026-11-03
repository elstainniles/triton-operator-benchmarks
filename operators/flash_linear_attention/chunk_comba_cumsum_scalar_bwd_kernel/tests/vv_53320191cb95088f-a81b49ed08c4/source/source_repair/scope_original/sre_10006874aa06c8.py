# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import triton

import triton.language as tl

from sre_74f55558e08fc1 import autotune_cache_kwargs

@triton.heuristics({
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
def chunk_comba_cumsum_scalar_bwd_kernel(
    dg0,
    dgr,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
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

    p_dg0 = tl.make_block_ptr(dg0 + bos*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_dgr = tl.make_block_ptr(dgr + bos*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    # [BT]
    """
    b_dg:   1,2,3,4
    b_dg0:  0,1,2,3
    b_temp: 0,1,3,6
    b_dz:   6
    b_dgr:  6,5,3,0
    """
    b_dg0 = tl.load(p_dg0, boundary_check=(0,)).to(tl.float32)
    b_temp = tl.cumsum(b_dg0, axis=0)
    b_dz = tl.sum(b_dg0, axis=0)
    b_dgr = -b_temp + b_dz[None]
    tl.store(p_dgr, b_dgr.to(p_dgr.dtype.element_ty), boundary_check=(0,))
