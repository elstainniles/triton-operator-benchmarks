# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: tests/ops/ascend_ops/vv_operator/k_update_fwd_kernel_long/test_k_update_fwd_kernel_long_npu.py:22 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
# Source archive: dependencies and launch adaptation are not executable here.
# The complete original file and dependency snapshots are under upstream/.
import triton
import triton.language as tl


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=w, num_stages=s)
        for w in NUM_WARPS_AUTOTUNE
        for s in [1, 2, 3]
    ],
    key=['BD', 'BT'],
)
@triton.jit
def k_update_fwd_kernel_long(
    k, a, ka, out,
    cu_seqlens, chunk_indices,
    T, D,
    BD: tl.constexpr, BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_d, i_t_blk, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    if IS_VARLEN:
        i_n, i_t_blk = tl.load(chunk_indices + i_t_blk * 2).to(tl.int32), \
            tl.load(chunk_indices + i_t_blk * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        t_start = i_t_blk * BT
        t_end = tl.minimum(t_start + BT, eos - bos)
    else:
        bos = i_b * T
        eos = (i_b + 1) * T
        t_start = i_t_blk * BT
        t_end = tl.minimum(t_start + BT, T)

    o_d = i_d * BD + tl.arange(0, BD)
    m_d = o_d < D

    for t in range(t_start, t_end):
        global_t = bos + t
        off = global_t * D + o_d
        b_k = tl.load(k + off, mask=m_d, other=0.).to(tl.float32)
        b_a = tl.load(a + off, mask=m_d, other=0.).to(tl.float32)
        b_ka = tl.load(ka + o_d, mask=m_d, eviction_policy='evict_last').to(tl.float32)
        out_val = b_k * (1 + (b_a - 1) * b_ka)
        tl.store(out + off, out_val.to(out.dtype.element_ty), mask=m_d)
