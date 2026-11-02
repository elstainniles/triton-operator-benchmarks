# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Fixed CV launch adapter; JIT signatures and bodies match upstream/.
# Autotune/heuristics are bound explicitly by cases-cv.json.
# Generated from exact archived source; function bodies unchanged.
import triton
import triton.language as tl
import math

@triton.jit
def softmax_bwd_kernel(p, dp, ds, D: tl.constexpr, B: tl.constexpr):
    i_n = tl.program_id(0)
    o_d = tl.arange(0, B)
    m_d = o_d < D
    b_p = tl.load(p + i_n * D + o_d, mask=m_d, other=0.0)
    b_dp = tl.load(dp + i_n * D + o_d, mask=m_d, other=0.0)
    b_pp = tl.sum(b_p * b_dp, 0)
    b_ds = b_p * b_dp - b_p * b_pp
    tl.store(ds + i_n * D + o_d, b_ds.to(ds.dtype.element_ty), mask=m_d)
