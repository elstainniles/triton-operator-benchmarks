# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors


"""Pinned FLA kernel; body/signature/JIT decorator unchanged.
Only autotune is removed; launch fixes an original candidate.
exp/log use the original FLA_USE_FAST_OPS=0 branch.
"""
import triton
import triton.language as tl



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
