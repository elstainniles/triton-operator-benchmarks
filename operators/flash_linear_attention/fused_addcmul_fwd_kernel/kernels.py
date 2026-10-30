# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: fla/ops/rwkv7/fused_addcmul.py:55 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
# Source archive: dependencies and launch adaptation are not executable here.
# The complete original file and dependency snapshots are under upstream/.
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BT': BT},  num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS_AUTOTUNE
        for num_stages in [1, 2, 3]
        for BT in [2, 4, 8]
    ],
    key=['BD'],
    use_cuda_graph=USE_CUDA_GRAPH,
    **autotune_cache_kwargs,
)
@triton.jit
def fused_addcmul_fwd_kernel(
    hidden,
    delta,
    ixr, ixw, ixk, ixv, ixa, ixg,
    oxr, oxw, oxk, oxv, oxa, oxg,
    use_xg: tl.constexpr,
    T,
    T_OFFSET,
    BT: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
):
    i_b, i_t = tl.program_id(0), tl.program_id(1) * BT

    bos = i_b * (T + T_OFFSET)
    t_vec = i_t + T_OFFSET + tl.arange(0, BT)
    mask_t = t_vec < (T + T_OFFSET)
    o_d = tl.arange(0, BD)[None, :]
    off_vec = (bos + t_vec)[:, None] * D + o_d
    m_d = o_d < D
    mask = mask_t[:, None] & m_d

    b_h = tl.load(hidden + off_vec, mask=mask, other=0.)
    b_x = tl.load(delta + off_vec, mask=mask, other=0.)
    b_r = tl.load(ixr + o_d, mask=m_d)
    b_w = tl.load(ixw + o_d, mask=m_d)
    b_k = tl.load(ixk + o_d, mask=m_d)
    b_v = tl.load(ixv + o_d, mask=m_d)
    b_a = tl.load(ixa + o_d, mask=m_d)

    o_r = tl.fma(b_x, b_r, b_h)
    o_w = tl.fma(b_x, b_w, b_h)
    o_k = tl.fma(b_x, b_k, b_h)
    o_v = tl.fma(b_x, b_v, b_h)
    o_a = tl.fma(b_x, b_a, b_h)

    tl.store(oxr + off_vec, o_r.to(oxr.dtype.element_ty), mask=mask)
    tl.store(oxw + off_vec, o_w.to(oxw.dtype.element_ty), mask=mask)
    tl.store(oxk + off_vec, o_k.to(oxk.dtype.element_ty), mask=mask)
    tl.store(oxv + off_vec, o_v.to(oxv.dtype.element_ty), mask=mask)
    tl.store(oxa + off_vec, o_a.to(oxa.dtype.element_ty), mask=mask)

    if use_xg:
        b_g = tl.load(ixg + o_d, mask=m_d)
        o_g = tl.fma(b_x, b_g, b_h)
        tl.store(oxg + off_vec, o_g.to(oxg.dtype.element_ty), mask=mask)
