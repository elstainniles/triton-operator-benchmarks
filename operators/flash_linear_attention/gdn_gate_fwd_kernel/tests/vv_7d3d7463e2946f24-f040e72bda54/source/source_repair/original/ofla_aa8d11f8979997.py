# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import triton

import triton.language as tl

from ofla_ab84b57b842c37 import fla_cache_autotune

from ofla_8459c530c3d6be import exp

from ofla_90891058742c80 import softplus

from ofla_74f55558e08fc1 import autocast_custom_bwd
from ofla_74f55558e08fc1 import autocast_custom_fwd
from ofla_74f55558e08fc1 import autotune_cache_kwargs
from ofla_74f55558e08fc1 import input_guard

@triton.heuristics({
    'HAS_BIAS': lambda args: args['dt_bias'] is not None,
})
@fla_cache_autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8]
    ],
    key=['H', 'BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def gdn_gate_bwd_kernel(
    g,
    A_log,
    dt_bias,
    dyg,
    dg,
    dA,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    i_t, i_h = tl.program_id(0), tl.program_id(1)

    b_A = tl.load(A_log + i_h).to(tl.float32)

    p_g = tl.make_block_ptr(g + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_dg = tl.make_block_ptr(dg + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_dyg = tl.make_block_ptr(dyg + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))

    b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
    b_dyg = tl.load(p_dyg, boundary_check=(0,)).to(tl.float32)

    if HAS_BIAS:
        b_g = b_g + tl.load(dt_bias + i_h).to(tl.float32)

    # gate = -exp(A_log) * softplus(g + bias)
    # d(gate)/d(g) = -exp(A_log) * sigmoid(g + bias)   (softplus' = sigmoid)
    # d(gate)/d(A_log) = -exp(A_log) * softplus(g + bias) = gate
    b_neg_expA = -exp(b_A)
    b_yg = b_neg_expA * softplus(b_g)
    b_dg = b_neg_expA * (b_dyg * tl.sigmoid(b_g))
    b_dA = tl.sum(b_dyg * b_yg, 0)

    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))
    tl.store(dA + i_t * H + i_h, b_dA)

@triton.heuristics({
    'HAS_BIAS': lambda args: args['dt_bias'] is not None,
})
@fla_cache_autotune(
    configs=[
        triton.Config({'BT': BT}, num_warps=num_warps, num_stages=num_stages)
        for BT in [32, 64, 128]
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3]
    ],
    key=['H'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def gdn_gate_fwd_kernel(
    g,
    A_log,
    dt_bias,
    yg,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    i_t, i_h = tl.program_id(0), tl.program_id(1)

    b_A = tl.load(A_log + i_h).to(tl.float32)

    p_g = tl.make_block_ptr(g + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_yg = tl.make_block_ptr(yg + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
    if HAS_BIAS:
        b_g = b_g + tl.load(dt_bias + i_h).to(tl.float32)
    b_yg = -exp(b_A) * softplus(b_g)
    tl.store(p_yg, b_yg.to(p_yg.dtype.element_ty), boundary_check=(0,))
