# Copyright 2024 LinkedIn Corporation. BSD-2-Clause; see ../LICENSE.
# Exact function/decorators extracted from the pinned upstream module.
# Loaded in that audited module namespace by opbench.liger; not a standalone import.
@triton.jit
def _softmax_single_block_backward_kernel(
    dy_ptr,
    dy_stride,
    y_ptr,
    y_stride,
    dx_ptr,
    dx_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols

    dy = tl.load(dy_ptr + row_id * dy_stride + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + row_id * y_stride + offs, mask=mask, other=0.0, cache_modifier=".ca")
    dot = tl.sum(dy * y, axis=0)
    dx = y * (dy - dot)
    tl.store(dx_ptr + row_id * dx_stride + offs, dx, mask=mask, cache_modifier=".wb")
