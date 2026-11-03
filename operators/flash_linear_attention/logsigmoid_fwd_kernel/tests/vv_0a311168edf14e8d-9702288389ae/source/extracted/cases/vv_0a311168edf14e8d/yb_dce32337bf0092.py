# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch

import triton

import triton.language as tl

from yb_8459c530c3d6be import exp
from yb_8459c530c3d6be import log

from yb_74f55558e08fc1 import IS_AMD
from yb_74f55558e08fc1 import autocast_custom_bwd
from yb_74f55558e08fc1 import autocast_custom_fwd
from yb_74f55558e08fc1 import autotune_cache_kwargs
from yb_74f55558e08fc1 import input_guard

NUM_WARPS_AUTOTUNE = [1, 2, 4, 8, 16] if IS_AMD else [1, 2, 4, 8, 16, 32]

def _get_stride(x: torch.Tensor) -> int:
    """Get the row stride for viewing a tensor as 2D (num_rows, D) where D = shape[-1].

    Returns stride(-2) if the tensor is at least 2D, or 0 for 1D tensors.
    The caller must ensure the tensor is "inner-contiguous" (stride(-1) == 1 and
    higher dims are contiguous relative to dim -2) before using this value.
    """
    if x.ndim < 2:
        return 0
    return x.stride(-2)

def _is_inner_contiguous(x: torch.Tensor) -> bool:
    """Check if a tensor can be safely viewed as 2D (num_rows, D) with row stride = stride(-2).

    This holds when stride(-1) == 1 and all dimensions above -2 are contiguous
    with respect to the dimension below them.
    """
    ndim = x.ndim
    if ndim < 2:
        return True
    if x.stride(-1) != 1:
        return False
    if ndim == 2:
        # 2D: any layout with stride(-1)==1 is valid (can view as (T, D))
        return True
    if ndim == 3:
        # 3D (B, T, D): stride should be (T*D, D, 1)
        return x.stride(0) == x.stride(-2) * x.shape[-2]
    if ndim == 4:
        # 4D (B, H, T, D): stride should be (H*T*D, T*D, D, 1)
        if x.stride(1) != x.stride(-2) * x.shape[-2]:
            return False
        return x.stride(0) == x.stride(1) * x.shape[1]
    # 5D+ fallback to loop
    expected = x.stride(-2) * x.shape[-2]
    for d in range(ndim - 3, -1, -1):
        if x.stride(d) != expected:
            return False
        expected *= x.shape[d]
    return True

def _ensure_inner_contiguous(x: torch.Tensor) -> torch.Tensor:
    """Make the tensor inner-contiguous if it isn't already."""
    if _is_inner_contiguous(x):
        return x
    return x.contiguous()

def _alloc_output(x: torch.Tensor, contiguous: bool = False) -> torch.Tensor:
    """Allocate output tensor: contiguous buffer or same layout as input."""
    if contiguous:
        return x.new_empty(x.shape)
    return torch.empty_like(x)

@triton.autotune(
    configs=[
        triton.Config({'B': bs}, num_warps=num_warps)
        for bs in [512, 1024, 2048, 4096, 8192]
        for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=['D'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def logsigmoid_fwd_kernel(
    x,
    y,
    temperature,
    T,
    D: tl.constexpr,
    stride_x_row,
    stride_y_row,
    B: tl.constexpr,
):
    i = tl.program_id(0)
    o_i = i * B + tl.arange(0, B)
    m_i = o_i < T
    row = o_i // D
    col = o_i % D
    x_off = row * stride_x_row + col
    y_off = row * stride_y_row + col

    b_x = tl.load(x + x_off, mask=m_i, other=0.).to(tl.float32)
    b_m = tl.minimum(0., b_x)
    b_z = 1. + exp(-tl.abs(b_x))
    b_y = (b_m - log(b_z)) / temperature
    tl.store(y + y_off, b_y.to(y.dtype.element_ty), mask=m_i)

@torch.compiler.disable
def logsigmoid_fwd(x: torch.Tensor, temperature: float = 1., output_contiguous: bool = False) -> torch.Tensor:
    x = _ensure_inner_contiguous(x)
    T, D = x.numel(), x.shape[-1]
    y = _alloc_output(x, output_contiguous)
    logsigmoid_fwd_kernel[lambda meta: (triton.cdiv(T, meta['B']),)](
        x=x,
        y=y,
        temperature=temperature,
        T=T,
        D=D,
        stride_x_row=_get_stride(x),
        stride_y_row=_get_stride(y),
    )
    return y
