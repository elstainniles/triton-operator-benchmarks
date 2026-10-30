# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: tests/ops/ascend_ops/vv_operator/chunk_comba_cumsum_scalar_fwd_kernel/test_chunk_comba_cumsum_scalar_fwd_kernel_npu.py:22 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
# Source archive: dependencies and launch adaptation are not executable here.
# The complete original file and dependency snapshots are under upstream/.
import triton
import triton.language as tl


@triton.heuristics({
    'HAS_SCALE': lambda args: args['scale'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8]
    ],
    key=['B', 'H', 'BT', 'IS_VARLEN'],
)
@triton.jit(do_not_specialize=['T'])
def chunk_comba_cumsum_scalar_fwd_kernel(
    g,
    g0,
    g1,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    p_g = tl.make_block_ptr(g + bos*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_g0 = tl.make_block_ptr(g0 + bos*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_g1 = tl.make_block_ptr(g1 + bos*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
    if HAS_SCALE:
        b_g = b_g * scale
    b_g1 = tl.cumsum(b_g, axis=0)
    b_g0 = b_g1 - b_g
    tl.store(p_g0, b_g0.to(p_g0.dtype.element_ty), boundary_check=(0,))
    tl.store(p_g1, b_g1.to(p_g1.dtype.element_ty), boundary_check=(0,))
