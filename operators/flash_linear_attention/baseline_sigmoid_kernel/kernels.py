# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: tests/ops/ascend_ops/vv_operator/sigmoid/test_sigmoid_npu.py:149 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
# Source archive: dependencies and launch adaptation are not executable here.
# The complete original file and dependency snapshots are under upstream/.
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': BS}, num_warps=num_warps)
        for BS in [256, 512, 1024, 2048]
        for num_warps in [1, 2, 4, 8]
    ],
    key=['N'],
    **autotune_cache_kwargs,
)
@triton.jit
def baseline_sigmoid_kernel(
    input,
    output,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(input + offsets, mask=mask, other=0.0).to(tl.float32)
    y = x
    tl.store(output + offsets, y.to(output.dtype.element_ty), mask=mask)
