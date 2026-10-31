# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: tests/ops/ascend_ops/vv_operator/k_update_fwd_kernel_short/test_k_update_fwd_kernel_short_npu.py:22 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
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
    key=['BD'],
)
@triton.jit
def k_update_fwd_kernel_short(
    k, a, ka, out,
    cu_seqlens,
    T, D,
    BD: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_b, i_t = tl.program_id(0), tl.program_id(1)

    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_b).to(tl.int32)
        eos = tl.load(cu_seqlens + i_b + 1).to(tl.int32)
        g_t = bos + i_t
        if g_t >= eos:
            return
        offset = g_t * D
    else:
        g_t = i_t
        offset = i_b * T * D + g_t * D

    o_d = tl.arange(0, BD)
    m_d = o_d < D
    off = offset + o_d

    b_k = tl.load(k + off, mask=m_d, other=0.).to(tl.float32)
    b_a = tl.load(a + off, mask=m_d, other=0.).to(tl.float32)
    b_ka = tl.load(ka + o_d, mask=m_d, eviction_policy='evict_last').to(tl.float32)

    out_val = b_k * (1 + (b_a - 1) * b_ka)
    tl.store(out + off, out_val.to(out.dtype.element_ty), mask=m_d)
