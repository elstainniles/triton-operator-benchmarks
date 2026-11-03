# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch

import triton

import triton.language as tl

from yb_ab84b57b842c37 import fla_cache_autotune

from yb_4f3456615af619 import prepare_chunk_indices

from yb_8459c530c3d6be import exp

from yb_90891058742c80 import softplus

from yb_74f55558e08fc1 import autocast_custom_bwd
from yb_74f55558e08fc1 import autocast_custom_fwd
from yb_74f55558e08fc1 import autotune_cache_kwargs
from yb_74f55558e08fc1 import input_guard

@triton.heuristics({
    'HAS_BIAS': lambda args: args['dt_bias'] is not None,
    'HAS_SCALE': lambda args: args['scale'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@fla_cache_autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8]
    ],
    key=['H', 'BT', 'IS_VARLEN', 'REVERSE'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def gdn_gate_chunk_cumsum_scalar_kernel(
    g,
    A_log,
    dt_bias,
    o,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
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

    p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_o = tl.make_block_ptr(o + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))

    b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
    if HAS_BIAS:
        b_g = b_g + tl.load(dt_bias + i_h).to(tl.float32)
    b_A = tl.load(A_log + i_h).to(tl.float32)
    b_gate = -exp(b_A) * softplus(b_g)

    b_o = tl.cumsum(b_gate, axis=0)
    if REVERSE:
        b_z = tl.sum(b_gate, axis=0)
        b_o = -b_o + b_z[None] + b_gate
    if HAS_SCALE:
        b_o *= scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))

@input_guard
def gdn_gate_chunk_cumsum(
    g: torch.Tensor,
    A_log: torch.Tensor,
    chunk_size: int,
    scale: float = None,
    dt_bias: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    output_dtype: torch.dtype | None = torch.float,
) -> torch.Tensor:
    B, T, H = g.shape
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    o = torch.empty_like(g, dtype=output_dtype or g.dtype)
    gdn_gate_chunk_cumsum_scalar_kernel[(NT, B * H)](
        g=g,
        A_log=A_log,
        dt_bias=dt_bias,
        o=o,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        BT=BT,
        REVERSE=False,
    )
    return o
