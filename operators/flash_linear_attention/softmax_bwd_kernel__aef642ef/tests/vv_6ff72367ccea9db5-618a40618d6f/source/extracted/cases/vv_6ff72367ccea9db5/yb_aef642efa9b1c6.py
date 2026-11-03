# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch

import triton

import triton.language as tl

from yb_74f55558e08fc1 import IS_AMD
from yb_74f55558e08fc1 import autotune_cache_kwargs

NUM_WARPS_AUTOTUNE = [1, 2, 4, 8, 16] if IS_AMD else [1, 2, 4, 8, 16, 32]

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=['D'],
    **autotune_cache_kwargs,
)
@triton.jit
def softmax_bwd_kernel(
    p,
    dp,
    ds,
    D: tl.constexpr,
    B: tl.constexpr,
):
    i_n = tl.program_id(0)
    o_d = tl.arange(0, B)
    m_d = o_d < D

    b_p = tl.load(p + i_n * D + o_d, mask=m_d, other=0.)
    b_dp = tl.load(dp + i_n * D + o_d, mask=m_d, other=0.)
    b_pp = tl.sum(b_p * b_dp, 0)
    b_ds = b_p * b_dp - b_p * b_pp
    tl.store(ds + i_n * D + o_d, b_ds.to(ds.dtype.element_ty), mask=m_d)

def softmax_bwd(
    p: torch.Tensor,
    dp: torch.Tensor,
    dtype: torch.dtype | None = torch.float,
) -> torch.Tensor:
    shape = p.shape
    p = p.view(-1, p.shape[-1])
    ds = torch.empty_like(p, dtype=dtype)

    N, D = p.shape
    B = triton.next_power_of_2(D)
    softmax_bwd_kernel[(N,)](
        p=p,
        dp=dp,
        ds=ds,
        D=D,
        B=B,
    )
    return ds.view(*shape)
