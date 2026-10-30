# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: tests/ops/ascend_ops/cv_operator/merge_fwd_bwd_kernel/test_merge_fwd_bwd_kernel_npu.py:29 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
# Source archive: dependencies and launch adaptation are not executable here.
# The complete original file and dependency snapshots are under upstream/.
import triton
import triton.language as tl


@triton.heuristics({
    'HAS_H0': lambda args: args['h0'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [2, 3, 4]
        for BV in [32, 64]
    ],
    key=['HV', 'K', 'V', 'BT'],
    use_cuda_graph=USE_CUDA_GRAPH,
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['pre_or_post_num_ranks', 'rank', 'NUM_SEQ_ENTRIES'])
def merge_fwd_bwd_kernel(
    h,
    ag_hm,
    pre_or_post_num_ranks,
    rank,
    seq_offsets,
    init_offsets,
    h0_seq_ids,
    h0,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    BK: tl.constexpr,
    FORWARD: tl.constexpr,
    INTRACARD_MODE: tl.constexpr,
    NUM_SEQ_ENTRIES,
    HAS_H0: tl.constexpr,
    TRANSPOSE_STATE: tl.constexpr = False,
):
    i_v = tl.program_id(0)
    if INTRACARD_MODE:
        i_seq = tl.program_id(1)
        i_h = tl.program_id(2)

        if i_seq >= NUM_SEQ_ENTRIES:
            return

        ss_start = tl.load(seq_offsets + i_seq).to(tl.int32)
        ss_end = tl.load(seq_offsets + i_seq + 1).to(tl.int32)
        init_base = tl.load(init_offsets + i_seq).to(tl.int32)
        num_subseqs = ss_end - ss_start

        stride_hm_s = HV * K * (V + K)
        stride_hm_h = K * (V + K)

        if HAS_H0:
            orig_seq_id = tl.load(h0_seq_ids + i_seq).to(tl.int32)
            if TRANSPOSE_STATE:
                p_h0 = tl.make_block_ptr(
                    h0 + (orig_seq_id * HV + i_h) * V * K,
                    (V, K), (K, 1), (i_v * BV, 0), (BV, BK), (1, 0)
                )
                b_h = tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)
            else:
                p_h0 = tl.make_block_ptr(
                    h0 + (orig_seq_id * HV + i_h) * K * V,
                    (K, V), (V, 1), (0, i_v * BV), (BK, BV), (1, 0)
                )
                b_h = tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)
        else:
            if TRANSPOSE_STATE:
                b_h = tl.zeros([BV, BK], dtype=tl.float32)
            else:
                b_h = tl.zeros([BK, BV], dtype=tl.float32)

        for idx in range(num_subseqs):
            i_ss = ss_start + idx
            base = i_ss * stride_hm_s + i_h * stride_hm_h

            p_he = tl.make_block_ptr(
                ag_hm + base, (K, V), (V + K, 1), (0, i_v * BV), (BK, BV), (1, 0)
            )
            b_he = tl.load(p_he, boundary_check=(0, 1)).to(tl.float32)
            p_m = tl.make_block_ptr(
                ag_hm + base + V, (K, K), (V + K, 1), (0, 0), (BK, BK), (1, 0)
            )
            b_m = tl.load(p_m, boundary_check=(0, 1)).to(tl.float32)
            if TRANSPOSE_STATE:
                b_h = tl.dot(b_h.to(tl.float32), tl.trans(b_m)) + tl.trans(b_he)
            else:
                b_h = tl.dot(b_m.to(tl.float32), b_h.to(tl.float32)) + b_he.to(tl.float32)

            if idx < num_subseqs - 1:
                init_idx = init_base + idx
                stride_init = HV * K * V
                if TRANSPOSE_STATE:
                    p_out = tl.make_block_ptr(
                        h + init_idx * stride_init + i_h * V * K,
                        (V, K), (K, 1), (i_v * BV, 0), (BV, BK), (1, 0)
                    )
                else:
                    p_out = tl.make_block_ptr(
                        h + init_idx * stride_init + i_h * K * V,
                        (K, V), (V, 1), (0, i_v * BV), (BK, BV), (1, 0)
                    )
                tl.store(p_out, b_h.to(p_out.dtype.element_ty), boundary_check=(0, 1))
    else:
        i_h = tl.program_id(1)
        num_ranks = pre_or_post_num_ranks.to(tl.int32)
        h += i_h * K * V
        ag_hm += i_h * K * (K + V)
        stride = HV * K * (K + V)
        if TRANSPOSE_STATE:
            b_h = tl.zeros([BV, BK], dtype=tl.float32)
        else:
            b_h = tl.zeros([BK, BV], dtype=tl.float32)
        for idx in range(num_ranks):
            if FORWARD:
                cur_rank = rank - num_ranks + idx
            else:
                cur_rank = rank + num_ranks - idx
            p_ag_h = tl.make_block_ptr(ag_hm + cur_rank * stride, (K, V), (K + V, 1), (0, i_v * BV), (BK, BV), (1, 0))
            b_ag_h = tl.load(p_ag_h, boundary_check=(0, 1))
            p_ag_m = tl.make_block_ptr(ag_hm + cur_rank * stride + V, (K, K), (K + V, 1), (0, 0), (BK, BK), (1, 0))
            b_ag_m = tl.load(p_ag_m, boundary_check=(0, 1))
            if TRANSPOSE_STATE:
                b_h = tl.dot(b_h.to(tl.float32), tl.trans(b_ag_m).to(tl.float32)) + tl.trans(b_ag_h).to(tl.float32)
            else:
                b_h = tl.dot(b_ag_m.to(tl.float32), b_h.to(tl.float32)) + b_ag_h.to(tl.float32)
        if TRANSPOSE_STATE:
            p_h = tl.make_block_ptr(h, (V, K), (K, 1), (i_v * BV, 0), (BV, BK), (1, 0))
        else:
            p_h = tl.make_block_ptr(h, (K, V), (V, 1), (0, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))
