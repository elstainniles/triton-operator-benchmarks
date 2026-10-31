# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: tests/ops/ascend_ops/vv_operator/layer_norm_gated_bwd_kernel1/test_layer_norm_gated_bwd_kernel1_npu.py:42 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
# Source archive: dependencies and launch adaptation are not executable here.
# The complete original file and dependency snapshots are under upstream/.
import triton
import triton.language as tl


@triton.heuristics(
    {
        "HAS_DRESIDUAL": lambda args: args["dresidual"] is not None,
        "HAS_WEIGHT": lambda args: args["w"] is not None,
        "HAS_BIAS": lambda args: args["b"] is not None,
        "RECOMPUTE_OUTPUT": lambda args: args["y"] is not None,
    }
)
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [2, 4, 8, 16]],
    key=["D", "IS_RMS_NORM", "STORE_DRESIDUAL", "HAS_DRESIDUAL", "HAS_WEIGHT"],
    **autotune_cache_kwargs,
)
@triton.jit
def layer_norm_gated_bwd_kernel1(
    x,
    g,
    w,
    b,
    y,
    dy,
    dx,
    dg,
    dw,
    db,
    dresidual,
    dresidual_in,
    mean,
    rstd,
    T,
    BS,
    D: tl.constexpr,
    BD: tl.constexpr,
    ACTIVATION: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    STORE_DRESIDUAL: tl.constexpr,
    HAS_DRESIDUAL: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    RECOMPUTE_OUTPUT: tl.constexpr,
):
    i_s = tl.program_id(0)
    o_d = tl.arange(0, BD)
    mask = o_d < D
    x += i_s * BS * D
    g += i_s * BS * D
    if HAS_DRESIDUAL:
        dresidual += i_s * BS * D
    if STORE_DRESIDUAL:
        dresidual_in += i_s * BS * D
    dy += i_s * BS * D
    dx += i_s * BS * D
    dg += i_s * BS * D
    if RECOMPUTE_OUTPUT:
        y += i_s * BS * D
    if HAS_WEIGHT:
        b_w = tl.load(w + o_d, mask=mask).to(tl.float32)
        b_dw = tl.zeros((BD,), dtype=tl.float32)
    if HAS_BIAS:
        b_b = tl.load(b + o_d, mask=mask, other=0.0).to(tl.float32)
        b_db = tl.zeros((BD,), dtype=tl.float32)

    for i_t in range(i_s * BS, min(i_s * BS + BS, T)):
        b_x = tl.load(x + o_d, mask=mask, other=0).to(tl.float32)
        b_g = tl.load(g + o_d, mask=mask, other=0).to(tl.float32)
        b_dy = tl.load(dy + o_d, mask=mask, other=0).to(tl.float32)

        if not IS_RMS_NORM:
            b_mean = tl.load(mean + i_t)
        b_rstd = tl.load(rstd + i_t)
        b_xhat = (b_x - b_mean) * b_rstd if not IS_RMS_NORM else b_x * b_rstd
        b_xhat = tl.where(mask, b_xhat, 0.0)

        b_y = b_xhat * b_w if HAS_WEIGHT else b_xhat
        if HAS_BIAS:
            b_y = b_y + b_b
        if RECOMPUTE_OUTPUT:
            tl.store(y + o_d, b_y, mask=mask)

        b_sigmoid_g = tl.sigmoid(b_g)
        if ACTIVATION == "swish" or ACTIVATION == "silu":
            b_dg = b_dy * b_y * (b_sigmoid_g + b_g * b_sigmoid_g * (1 - b_sigmoid_g))
            b_dy = b_dy * b_g * b_sigmoid_g
        elif ACTIVATION == "sigmoid":
            b_dg = b_dy * b_y * b_sigmoid_g * (1 - b_sigmoid_g)
            b_dy = b_dy * b_sigmoid_g
        b_wdy = b_dy
        if HAS_WEIGHT:
            b_wdy = b_dy * b_w
            b_dw += b_dy * b_xhat
        if HAS_BIAS:
            b_db += b_dy
        if not IS_RMS_NORM:
            b_c1 = tl.sum(b_xhat * b_wdy, axis=0) / D
            b_c2 = tl.sum(b_wdy, axis=0) / D
            b_dx = (b_wdy - (b_xhat * b_c1 + b_c2)) * b_rstd
        else:
            b_c1 = tl.sum(b_xhat * b_wdy, axis=0) / D
            b_dx = (b_wdy - b_xhat * b_c1) * b_rstd
        if HAS_DRESIDUAL:
            b_dres = tl.load(dresidual + o_d, mask=mask, other=0).to(tl.float32)
            b_dx += b_dres
        if STORE_DRESIDUAL:
            tl.store(dresidual_in + o_d, b_dx, mask=mask)
        tl.store(dx + o_d, b_dx, mask=mask)
        tl.store(dg + o_d, b_dg, mask=mask)

        x += D
        g += D
        if HAS_DRESIDUAL:
            dresidual += D
        if STORE_DRESIDUAL:
            dresidual_in += D
        if RECOMPUTE_OUTPUT:
            y += D
        dy += D
        dx += D
        dg += D
    if HAS_WEIGHT:
        tl.store(dw + i_s * D + o_d, b_dw, mask=mask)
    if HAS_BIAS:
        tl.store(db + i_s * D + o_d, b_db, mask=mask)
