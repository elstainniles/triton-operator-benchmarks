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
def exp(x): return tl.exp(x.to(tl.float32))

@triton.jit
def softmax_fwd_kernel(
    x,
    p,
    D: tl.constexpr,
    B: tl.constexpr,
):
    i_n = tl.program_id(0)
    o_d = tl.arange(0, B)
    m_d = o_d < D

    b_x = tl.load(x + i_n * D + o_d, mask=m_d, other=-float('inf'))
    b_m = tl.max(b_x, 0)
    b_x = exp(b_x - b_m)
    b_p = b_x / tl.sum(b_x, 0)

    tl.store(p + i_n * D + o_d, b_p.to(p.dtype.element_ty), mask=m_d)
