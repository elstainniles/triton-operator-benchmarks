# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# This file is modified and supported by the Moonshot AI Team

import torch

import triton

import triton.language as tl

from yb_ab84b57b842c37 import fla_cache_autotune

from yb_8459c530c3d6be import exp

from yb_90891058742c80 import softplus

from yb_74f55558e08fc1 import IS_AMD
from yb_74f55558e08fc1 import autocast_custom_bwd
from yb_74f55558e08fc1 import autocast_custom_fwd
from yb_74f55558e08fc1 import autotune_cache_kwargs
from yb_74f55558e08fc1 import check_shared_mem
from yb_74f55558e08fc1 import input_guard

NUM_WARPS_AUTOTUNE = [2, 4, 8, 16] if IS_AMD else [4, 8, 16, 32]

@triton.heuristics({
    "HAS_BIAS": lambda args: args["dt_bias"] is not None,
    "HAS_BETA": lambda args: args["beta"] is not None,
    'USE_LOWER_BOUND': lambda args: args['lower_bound'] is not None,
})
@fla_cache_autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS_AUTOTUNE
        for num_stages in [2, 3]
    ],
    key=["H", "D"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def kda_gate_bwd_kernel(
    g,
    A_log,
    dt_bias,
    beta,
    dyg,
    dyb,
    dg,
    dA,
    dbeta,
    lower_bound,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_BETA: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
):
    i_t, i_h = tl.program_id(0), tl.program_id(1)

    b_A = tl.load(A_log + i_h).to(tl.float32)

    p_g = tl.make_block_ptr(g + i_h * D, (T, D), (H * D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    p_dg = tl.make_block_ptr(dg + i_h * D, (T, D), (H * D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    p_dyg = tl.make_block_ptr(dyg + i_h * D, (T, D), (H * D, 1), (i_t * BT, 0), (BT, BD), (1, 0))

    # [BT, BD]
    b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
    b_dyg = tl.load(p_dyg, boundary_check=(0, 1)).to(tl.float32)

    if HAS_BIAS:
        p_b = tl.make_block_ptr(dt_bias, (H * D,), (1,), (i_h * D,), (BD,), (0,))
        b_g = b_g + tl.load(p_b, boundary_check=(0,)).to(tl.float32)

    # [BT, BD]
    if not USE_LOWER_BOUND:
        b_A = -exp(b_A)
        b_yg = b_A * softplus(b_g)
        b_dg = b_A * (b_dyg * tl.sigmoid(b_g))
        b_dA = tl.sum(tl.sum(b_dyg * b_yg, 1), 0)
    else:
        b_A = exp(b_A)
        b_inner = b_A * b_g
        b_sig = tl.sigmoid(b_inner)
        b_dsig = b_sig * (1.0 - b_sig)
        # Common term: dy * (LB * dsig)
        b_d_inner_term = b_dyg * (lower_bound * b_dsig)
        # dg = d_inner_term * A
        b_dg = b_d_inner_term * b_A
        b_dA = tl.sum(tl.sum(b_dg * b_g, 1), 0)

    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0, 1))
    tl.store(dA + i_t * H + i_h, b_dA)

    if HAS_BETA:
        p_b = tl.make_block_ptr(beta + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        p_db = tl.make_block_ptr(dbeta + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        p_dyb = tl.make_block_ptr(dyb + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))

        b_b = tl.load(p_b, boundary_check=(0,)).to(tl.float32)
        b_db = tl.load(p_dyb, boundary_check=(0,)).to(tl.float32) * b_b * (1.0 - b_b)
        tl.store(p_db, b_db.to(p_db.dtype.element_ty), boundary_check=(0,))

def kda_gate_bwd(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None = None,
    dyg: torch.Tensor | None = None,
    lower_bound: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    H, K = g.shape[-2:]
    T = g.numel() // (H * K)
    BT = 32
    NT = triton.cdiv(T, BT)

    dg = torch.empty_like(g, dtype=torch.float32)
    dA = A_log.new_empty(NT, H, dtype=torch.float32)

    grid = (triton.cdiv(T, BT), H)
    kda_gate_bwd_kernel[grid](
        g=g,
        A_log=A_log,
        dt_bias=dt_bias,
        beta=None,
        dyg=dyg,
        dyb=None,
        dg=dg,
        dA=dA,
        dbeta=None,
        T=T,
        H=H,
        D=K,
        BT=BT,
        BD=triton.next_power_of_2(K),
        lower_bound=lower_bound,
    )

    dg = dg.view_as(g).type_as(g)
    dA = dA.sum(0).view_as(A_log).type_as(A_log)
    dbias = dg.view(-1, H * K).sum(0).to(dt_bias) if dt_bias is not None else None

    return dg, dA, dbias
