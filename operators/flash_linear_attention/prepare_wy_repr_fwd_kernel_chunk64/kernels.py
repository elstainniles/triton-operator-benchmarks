# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: tests/ops/ascend_ops/cv_operator/prepare_wy_repr_fwd_kernel_chunk64/test_prepare_wy_repr_fwd_kernel_chunk64_npu.py:37 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
# Source archive: dependencies and launch adaptation are not executable here.
# The complete original file and dependency snapshots are under upstream/.
import triton
import triton.language as tl


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({})
    ],
    key=['BC'],
    use_cuda_graph=USE_CUDA_GRAPH,
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def prepare_wy_repr_fwd_kernel_chunk64(
        A_ab,
        A_ab_inv,
        cu_seqlens,
        chunk_indices,
        T,
        H: tl.constexpr,
        BT: tl.constexpr,
        BC: tl.constexpr,
        IS_VARLEN: tl.constexpr,
        GATHER_SUPPORTED: tl.constexpr = IS_GATHER_SUPPORTED,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    p_A1 = tl.make_block_ptr(A_ab + (bos * H + i_h) * BT, (T, BT), (H * BT, 1), (i_t * BT, 0), (BC, BC), (1, 0))
    p_A2 = tl.make_block_ptr(A_ab + (bos * H + i_h) * BT, (T, BT), (H * BT, 1), (i_t * BT + BC, BC), (BC, BC), (1, 0))
    p_A3 = tl.make_block_ptr(A_ab + (bos * H + i_h) * BT, (T, BT), (H * BT, 1), (i_t * BT + BC, 0), (BC, BC), (1, 0))
    p_A_inv1 = tl.make_block_ptr(A_ab_inv + (bos * H + i_h) * BT, (T, BT), (H * BT, 1), (i_t * BT, 0), (BC, BC), (1, 0))
    p_A_inv2 = tl.make_block_ptr(A_ab_inv + (bos * H + i_h) * BT, (T, BT), (H * BT, 1), (i_t * BT + BC, BC), (BC, BC),
                                 (1, 0))
    p_A_inv3 = tl.make_block_ptr(A_ab_inv + (bos * H + i_h) * BT, (T, BT), (H * BT, 1), (i_t * BT + BC, 0), (BC, BC),
                                 (1, 0))
    p_A_inv4 = tl.make_block_ptr(A_ab_inv + (bos * H + i_h) * BT, (T, BT), (H * BT, 1), (i_t * BT, BC), (BC, BC),
                                 (1, 0))

    b_A = tl.load(p_A1, boundary_check=(0, 1))
    b_A2 = tl.load(p_A2, boundary_check=(0, 1))
    b_A3 = tl.load(p_A3, boundary_check=(0, 1))
    b_A = tl.where(tl.arange(0, BC)[:, None] > tl.arange(0, BC)[None, :], b_A, 0)
    b_A2 = tl.where(tl.arange(0, BC)[:, None] > tl.arange(0, BC)[None, :], b_A2, 0)

    for i in range(1, BC):
        if GATHER_SUPPORTED:
            row_idx = tl.full([1, BC], i, dtype=tl.int16)
            b_a = tl.sum(gather(b_A, row_idx, axis=0), 0)
            b_a2 = tl.sum(gather(b_A2, row_idx, axis=0), 0)
        else:
            mask = tl.arange(0, BC) == i
            b_a = tl.sum(tl.where(mask[:, None], b_A, 0), 0)
            b_a2 = tl.sum(tl.where(mask[:, None], b_A2, 0), 0)
        mask = tl.arange(0, BC) == i
        b_a = b_a + tl.sum(b_a[:, None] * b_A, 0) * (tl.arange(0, BC) < i)
        b_a2 = b_a2 + tl.sum(b_a2[:, None] * b_A2, 0) * (tl.arange(0, BC) < i)
        b_A = tl.where(mask[:, None], b_a, b_A)
        b_A2 = tl.where(mask[:, None], b_a2, b_A2)

    b_A += tl.arange(0, BC)[:, None] == tl.arange(0, BC)[None, :]
    b_A2 += tl.arange(0, BC)[:, None] == tl.arange(0, BC)[None, :]
    b_A3 = tl.dot(tl.dot(b_A2, b_A3), b_A)
    tl.store(p_A_inv1, b_A.to(p_A_inv1.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_A_inv2, b_A2.to(p_A_inv2.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_A_inv3, b_A3.to(p_A_inv3.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_A_inv4, tl.zeros([BC, BC], dtype=tl.float32).to(p_A_inv4.dtype.element_ty), boundary_check=(0, 1))
