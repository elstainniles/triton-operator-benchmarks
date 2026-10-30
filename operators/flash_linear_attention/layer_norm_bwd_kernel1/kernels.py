# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: fla/modules/layernorm.py:458 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
# Source archive: dependencies and launch adaptation are not executable here.
# The complete original file and dependency snapshots are under upstream/.
import triton
import triton.language as tl


@triton.heuristics({
    'RECOMPUTE_OUTPUT': lambda args: args['y'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [2, 4, 8]
    ],
    key=['D', 'HAS_DRESIDUAL', 'STORE_DRESIDUAL', 'IS_RMS_NORM'],
    **autotune_cache_kwargs,
)
@triton.jit
def layer_norm_bwd_kernel1(
    x,  # pointer to the input
    w,  # pointer to the weights
    b,  # pointer to the biases
    y,  # pointer to the output to be recomputed
    dy,  # pointer to the output gradient
    dx,  # pointer to the input gradient
    dw,  # pointer to the partial sum of weights gradient
    db,  # pointer to the partial sum of biases gradient
    dres,
    dres_in,
    mean,
    rstd,
    T,
    G: tl.constexpr,
    D: tl.constexpr,
    BS: tl.constexpr,
    BD: tl.constexpr,
    GS: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    HAS_DRESIDUAL: tl.constexpr,
    STORE_DRESIDUAL: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    RECOMPUTE_OUTPUT: tl.constexpr,
):
    i_s = tl.program_id(0)
    i_g, i_sg = i_s // GS, i_s % GS

    o_d = tl.arange(0, BD)
    mask = o_d < D

    if HAS_WEIGHT:
        b_w = tl.load(w + i_g * D + o_d, mask=mask).to(tl.float32)
        b_dw = tl.zeros((BD,), dtype=tl.float32)
    if RECOMPUTE_OUTPUT and HAS_BIAS:
        b_b = tl.load(b + i_g * D + o_d, mask=mask, other=0.0).to(tl.float32)
    if HAS_BIAS:
        b_db = tl.zeros((BD,), dtype=tl.float32)

    for i_t in range(i_sg * BS * G + i_g, min((i_sg * BS + BS) * G + i_g, T), G):
        b_x = tl.load(x + i_t * D + o_d, mask=mask, other=0).to(tl.float32)
        b_dy = tl.load(dy + i_t * D + o_d, mask=mask, other=0).to(tl.float32)

        if not IS_RMS_NORM:
            b_mean = tl.load(mean + i_t)
        b_rstd = tl.load(rstd + i_t)
        # Compute dx
        b_xhat = (b_x - b_mean) * b_rstd if not IS_RMS_NORM else b_x * b_rstd
        b_xhat = tl.where(mask, b_xhat, 0.0)
        if RECOMPUTE_OUTPUT:
            b_y = b_xhat * b_w if HAS_WEIGHT else b_xhat
            if HAS_BIAS:
                b_y = b_y + b_b
            tl.store(y + i_t * D + o_d, b_y, mask=mask)
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
            b_dres = tl.load(dres + i_t * D + o_d, mask=mask, other=0).to(tl.float32)
            b_dx += b_dres
        # Write dx
        b_dx = tl.cast(b_dx, dtype=dx.dtype.element_ty, fp_downcast_rounding='rtne')
        if STORE_DRESIDUAL:
            tl.store(dres_in + i_t * D + o_d, b_dx, mask=mask)
        tl.store(dx + i_t * D + o_d, b_dx, mask=mask)

    if HAS_WEIGHT:
        tl.store(dw + i_s * D + o_d, b_dw, mask=mask)
    if HAS_BIAS:
        tl.store(db + i_s * D + o_d, b_db, mask=mask)
