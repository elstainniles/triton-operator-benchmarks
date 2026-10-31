# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: tests/ops/ascend_ops/vv_operator/_compare_and_swap/test_compare_and_swap_npu.py:88 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
# Source archive: dependencies and launch adaptation are not executable here.
# The complete original file and dependency snapshots are under upstream/.
import triton
import triton.language as tl


@triton.jit
def compare_and_swap_kernel(
    x_ptr,
    ids_ptr,
    output_x_ptr,
    output_ids_ptr,
    N: tl.constexpr,
    i: tl.constexpr,
    n_dims: tl.constexpr,
):
    """
    直接调用_compare_and_swap函数的wrapper kernel
    """
    # 加载输入
    x = tl.load(x_ptr + tl.arange(0, N))
    ids = tl.load(ids_ptr + tl.arange(0, N))
    
    # 创建flip tensor（全0表示升序）
    flip = tl.zeros([N], dtype=tl.int1)
    
    # 直接调用_compare_and_swap
    x_new, ids_new = _compare_and_swap(x, ids, flip, i, n_dims)
    
    # 存储输出
    tl.store(output_x_ptr + tl.arange(0, N), x_new)
    tl.store(output_ids_ptr + tl.arange(0, N), ids_new)
