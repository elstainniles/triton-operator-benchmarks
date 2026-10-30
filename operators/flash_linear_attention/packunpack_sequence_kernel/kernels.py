# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: tests/ops/ascend_ops/vv_operator/packunpack_sequence_kernel/test_packunpack_sequence_kernel_npu.py:18 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
# Source archive: dependencies and launch adaptation are not executable here.
# The complete original file and dependency snapshots are under upstream/.
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [4, 8, 16, 32]
    ],
    key=['D', 'PADDING_SIDE', 'PACK'],
)
@triton.jit
def packunpack_sequence_kernel(
    x,
    y,
    cu_seqlens,
    S,
    D,
    BD: tl.constexpr,
    PADDING_SIDE: tl.constexpr,
    PACK: tl.constexpr,
):
    i_d, i_s, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    bos, eos = tl.load(cu_seqlens + i_b), tl.load(cu_seqlens + i_b + 1)

    T = eos - bos
    if PADDING_SIDE == 'left':
        NP = S - T
        if i_s < NP:
            return
        i_t = bos + (i_s - NP)
    else:
        if i_s >= T:
            return
        i_t = bos + i_s

    o_d = i_d * BD + tl.arange(0, BD)
    mask = o_d < D

    if PACK:
        b_x = tl.load(x + (i_b * S + i_s) * D + o_d, mask=mask)
        tl.store(y + i_t * D + o_d, b_x, mask=mask)
    else:
        b_x = tl.load(x + i_t * D + o_d, mask=mask)
        tl.store(y + (i_b * S + i_s) * D + o_d, b_x, mask=mask)
