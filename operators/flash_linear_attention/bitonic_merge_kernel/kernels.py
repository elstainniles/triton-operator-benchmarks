# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: tests/ops/ascend_ops/vv_operator/_bitonic_merge/test_bitonic_merge_npu.py:114 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
# Source archive: dependencies and launch adaptation are not executable here.
# The complete original file and dependency snapshots are under upstream/.
import triton
import triton.language as tl


@triton.jit
def bitonic_merge_kernel(
    x_ptr,
    ids_ptr,
    output_x_ptr,
    output_ids_ptr,
    N: tl.constexpr,
    stage: tl.constexpr,
    order: tl.constexpr,
    n_dims: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    直接调用_bitonic_merge函数的wrapper kernel
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    # 加载输入
    x = tl.load(x_ptr + offsets, mask=mask)
    ids = tl.load(ids_ptr + offsets, mask=mask)

    # 直接调用_bitonic_merge
    x_new, ids_new = _bitonic_merge(x, ids, stage, order, n_dims)
    
    # 存储输出
    tl.store(output_x_ptr + offsets, x_new, mask=mask)
    tl.store(output_ids_ptr + offsets, ids_new, mask=mask)
