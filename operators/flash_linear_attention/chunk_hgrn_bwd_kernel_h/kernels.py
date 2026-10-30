# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: tests/ops/ascend_ops/vv_operator/chunk_hgrn_bwd_kernel_h/test_chunk_hgrn_bwd_kernel_h_npu.py:208 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
# Source archive: dependencies and launch adaptation are not executable here.
# The complete original file and dependency snapshots are under upstream/.
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BD': BD}, num_warps=num_warps)
        for BD in [32, 64, 128]
        for num_warps in [1, 2, 4, 8]
    ],
    key=['D'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_hgrn_bwd_kernel_h(
    g,
    gc,
    dx,
    do,
    T,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    i_d, i_t, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_d = i_d * BD + tl.arange(0, BD)
    mask = o_d < D
    BC = min(BT, T - i_t * BT)
    NT = tl.num_programs(1)

    p_g = g + (i_b * T + i_t * BT + BC - 1) * D + o_d
    p_gc = gc + (i_b * T + i_t * BT + BC - 1) * D + o_d
    p_dx = dx + (i_b * T + i_t * BT + BC - 1) * D + o_d
    p_do = do + (i_b * T + i_t * BT + BC - 1) * D + o_d

    if i_t == NT - 1:
        b_gc = tl.zeros([BD], dtype=tl.float32)
    else:
        b_gc = tl.load(g + (i_b * T + i_t * BT + BT) * D + o_d, mask=mask, other=0).to(tl.float32)
    b_dh = tl.zeros([BD], dtype=tl.float32)
    for _ in range(BC - 1, -1, -1):
        tl.store(p_gc, b_gc.to(p_gc.dtype.element_ty), mask=mask)

        b_g = tl.load(p_g, mask=mask, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask, other=0).to(tl.float32)

        b_gc = b_gc + b_g
        b_dh = b_dh + b_do
        b_dx = b_dh
        b_dh = b_dh * exp(b_g)

        tl.store(p_dx, b_dx.to(p_dx.dtype.element_ty), mask=mask)

        p_g -= D
        p_gc -= D
        p_dx -= D
        p_do -= D
