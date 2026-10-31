# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: fla/ops/rwkv7/channel_mixing.py:174 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
# Source archive: dependencies and launch adaptation are not executable here.
# The complete original file and dependency snapshots are under upstream/.
import triton
import triton.language as tl


@triton.jit
def relu_square_bwd_kernel(
    out_ptr,
    forward_input_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    """ReLU(x)^2 backward kernel
    grad_input = grad_output * 2 * x if x > 0 else 0
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    x = tl.load(forward_input_ptr + offsets).to(tl.float32)
    grad = tl.load(out_ptr + offsets).to(tl.float32)

    x = tl.maximum(x, 0.0)

    grad_input = grad * 2 * x

    tl.store(out_ptr + offsets, grad_input.to(out_ptr.dtype.element_ty))
