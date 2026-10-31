# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: fla/ops/rwkv7/fused_addcmul.py:115 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
# Source archive: dependencies and launch adaptation are not executable here.
# The complete original file and dependency snapshots are under upstream/.
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BT': BT}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS_AUTOTUNE
        for num_stages in [1, 2, 3]
        for BT in [2, 4, 8]
    ],
    key=['BD'],
    use_cuda_graph=USE_CUDA_GRAPH,
    **autotune_cache_kwargs,
)
@triton.jit
def addcmul_bwd_kernel1(
    ixr,
    ixw,
    ixk,
    ixv,
    ixa,
    ixg,
    dxr,
    dxw,
    dxk,
    dxv,
    dxa,
    dxg,
    ghidden,
    gx,
    use_xg: tl.constexpr,
    T,
    T_OFFSET,
    BT: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    DTYPE: tl.constexpr,
):
    i_b, i_t = tl.program_id(0), tl.program_id(1)

    t_idx = T_OFFSET + i_t * BT + tl.arange(0, BT)[:, None]
    mask_t = t_idx < (T + T_OFFSET)

    d_idx = tl.arange(0, BD)[None, :]
    mask_d = d_idx < D
    mask = mask_t & mask_d

    offset_base = i_b * (T + T_OFFSET) * D
    x_idx = (offset_base + t_idx * D + d_idx).to(tl.uint32)

    b_dxr = tl.load(dxr + x_idx, mask=mask).to(DTYPE)
    b_dxw = tl.load(dxw + x_idx, mask=mask).to(DTYPE)
    b_dxk = tl.load(dxk + x_idx, mask=mask).to(DTYPE)
    b_dxv = tl.load(dxv + x_idx, mask=mask).to(DTYPE)
    b_dxa = tl.load(dxa + x_idx, mask=mask).to(DTYPE)

    b_ixr = tl.load(ixr + d_idx, mask=mask_d).to(DTYPE)
    b_ixw = tl.load(ixw + d_idx, mask=mask_d).to(DTYPE)
    b_ixk = tl.load(ixk + d_idx, mask=mask_d).to(DTYPE)
    b_ixv = tl.load(ixv + d_idx, mask=mask_d).to(DTYPE)
    b_ixa = tl.load(ixa + d_idx, mask=mask_d).to(DTYPE)

    g_hidden = b_dxr + b_dxw + b_dxk + b_dxv + b_dxa
    g_x = b_dxr * b_ixr + b_dxw * b_ixw + b_dxk * b_ixk + b_dxv * b_ixv + b_dxa * b_ixa

    if use_xg:
        b_dxg = tl.load(dxg + x_idx, mask=mask).to(DTYPE)
        b_ixg = tl.load(ixg + d_idx, mask=mask_d).to(DTYPE)
        g_hidden += b_dxg
        g_x += b_dxg * b_ixg

    tl.store(ghidden + x_idx, g_hidden.to(ghidden.dtype.element_ty), mask=mask)
    tl.store(gx + x_idx, g_x.to(gx.dtype.element_ty), mask=mask)
