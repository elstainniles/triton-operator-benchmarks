# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch

import triton

import triton.language as tl

from yb_4f3456615af619 import prepare_chunk_indices

from yb_74f55558e08fc1 import autocast_custom_bwd
from yb_74f55558e08fc1 import autocast_custom_fwd
from yb_74f55558e08fc1 import autotune_cache_kwargs
from yb_74f55558e08fc1 import input_guard

@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BD': BD}, num_warps=num_warps)
        for BD in [16, 32, 64, 128]
        for num_warps in [1, 2, 4, 8]
    ],
    key=['BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def mean_pooling_bwd_kernel(
    do,
    dx,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_d, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    p_dx = tl.make_block_ptr(dx + (bos * H + i_h) * D, (T, D), (H*D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
    p_do = tl.make_block_ptr(do + (i_tg * H + i_h) * D, (D,), (1,), (i_d * BD,), (BD,), (0,))
    # [BD]
    b_do = tl.load(p_do, boundary_check=(0,)).to(tl.float32)
    # [BT, BD]
    b_dx = b_do / tl.full((BT,), min(BT, T - i_t * BT), dtype=tl.float32)[:, None]
    tl.store(p_dx, b_dx.to(p_dx.dtype.element_ty), boundary_check=(0, 1))

def mean_pooling_bwd(
    do: torch.Tensor,
    batch_size: int,
    seq_len: int,
    chunk_size: int,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> torch.Tensor:
    B, T, H, D = batch_size, seq_len, *do.shape[-2:]
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    dx = do.new_empty(B, T, H, D)
    def grid(meta): return (triton.cdiv(D, meta['BD']), NT, B * H)
    mean_pooling_bwd_kernel[grid](
        do,
        dx,
        cu_seqlens,
        chunk_indices,
        T=T,
        H=H,
        D=D,
        BT=BT,
    )
    return dx
