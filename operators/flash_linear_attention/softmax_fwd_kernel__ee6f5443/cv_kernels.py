# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Fixed CV launch adapter; JIT signatures and bodies match upstream/.
# Autotune/heuristics are bound explicitly by cases-cv.json.
# Generated from exact archived source; function bodies unchanged.
import triton
import triton.language as tl
import math

@triton.jit
def exp(x):
    return tl.exp(x.to(tl.float32))

@triton.jit
def softmax_fwd_kernel(x, p, D: tl.constexpr, B: tl.constexpr):
    i_n = tl.program_id(0)
    o_d = tl.arange(0, B)
    m_d = o_d < D
    b_x = tl.load(x + i_n * D + o_d, mask=m_d, other=-float('inf'))
    b_m = tl.max(b_x, 0)
    b_x = exp(b_x - b_m)
    b_p = b_x / tl.sum(b_x, 0)
    tl.store(p + i_n * D + o_d, b_p.to(p.dtype.element_ty), mask=m_d)
