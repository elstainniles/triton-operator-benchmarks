# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: fla/ops/rwkv7/channel_mixing.py:84 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
# Source archive: dependencies and launch adaptation are not executable here.
# The complete original file and dependency snapshots are under upstream/.
import triton
import triton.language as tl


@triton.jit
def rwkv_channel_mixing_pow_and_relu(
    in_ptr,
    out_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused ReLU and Power operation: x = ReLU(x)^2"""
    xoffset = tl.program_id(0) * BLOCK_SIZE
    xindex = xoffset + tl.arange(0, BLOCK_SIZE)
    x0 = xindex
    x = tl.load(in_ptr + (x0), None)
    x = tl.maximum(x, 0.0).to(tl.float32)
    x = tl.cast(x * x, dtype=out_ptr.dtype.element_ty, fp_downcast_rounding='rtne')
    tl.store(out_ptr + (x0), x, None)
