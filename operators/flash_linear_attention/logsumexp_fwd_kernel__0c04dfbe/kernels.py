# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: fla/modules/fused_linear_cross_entropy.py:43 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
# Source archive: dependencies and launch adaptation are not executable here.
# The complete original file and dependency snapshots are under upstream/.
import triton
import triton.language as tl


@triton.heuristics({
    'HAS_SCALE': lambda args: args['scale'] is not None,
    'HAS_SOFTCAPPING': lambda args: args['softcapping'] is not None,
})
@triton.jit
def logsumexp_fwd_kernel(
    x,
    z,
    scale,
    softcapping,
    D: tl.constexpr,
    B: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    HAS_SOFTCAPPING: tl.constexpr,
):
    i_n, i_d = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    o_d = i_d * B + tl.arange(0, B)
    m_d = o_d < D

    b_x = tl.load(x + i_n * D + o_d, mask=m_d, other=-float('inf'))
    if HAS_SCALE:
        b_x = b_x * scale
    if HAS_SOFTCAPPING:
        b_x = softcapping * tanh(b_x / softcapping)
    b_m = tl.max(b_x, 0)
    b_z = log(tl.sum(exp(b_x - b_m), 0)) + b_m
    tl.store(z + i_n * tl.cdiv(D, B) + i_d, b_z)
