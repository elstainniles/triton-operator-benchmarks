# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Fixed CV launch adapter; JIT signatures and bodies match upstream/.
# Autotune/heuristics are bound explicitly by cases-cv.json.
# Generated from exact archived source; function bodies unchanged.
import triton
import triton.language as tl
import math

@triton.jit
def safe_dot(a, b, allow_tf32: tl.constexpr=None):
    return tl.dot(a, b, allow_tf32=allow_tf32)

@triton.jit
def safe_dot_kernel(a_ptr, b_ptr, c_ptr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rm = tl.arange(0, BM)
    rn = tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a = tl.load(a_ptr + rm[:, None] * BK + rk[None, :])
    b = tl.load(b_ptr + rk[:, None] * BN + rn[None, :])
    tl.store(c_ptr + rm[:, None] * BN + rn[None, :], safe_dot(a, b, allow_tf32=False))
