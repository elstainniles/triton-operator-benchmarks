# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: tests/ops/ascend_ops/cv_operator/chunk_abc/chunk_abc.py:174 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
# Source archive: dependencies and launch adaptation are not executable here.
# The complete original file and dependency snapshots are under upstream/.
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8, 16, 32]
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
