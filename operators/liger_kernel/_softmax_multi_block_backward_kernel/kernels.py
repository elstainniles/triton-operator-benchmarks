# Copyright 2024 LinkedIn Corporation. BSD-2-Clause; see ../LICENSE.
# Exact function/decorators extracted from the pinned upstream module.
# Loaded in that audited module namespace by opbench.liger; not a standalone import.
@triton.jit
def _softmax_multi_block_backward_kernel(
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
    acc = tl.float32(0.0)

    for start in tl.range(0, n_cols, BLOCK_SIZE):
        idx = start + offs
        mask = idx < n_cols
        dy_blk = tl.load(dy_ptr + row_id * dy_stride + idx, mask=mask, other=0.0)
        y_blk = tl.load(y_ptr + row_id * y_stride + idx, mask=mask, other=0.0, cache_modifier=".ca")
        acc += tl.sum(dy_blk * y_blk, axis=0)

    for start in tl.range(0, n_cols, BLOCK_SIZE):
        idx = start + offs
        mask = idx < n_cols
        dy_blk = tl.load(dy_ptr + row_id * dy_stride + idx, mask=mask, other=0.0)
        y_blk = tl.load(y_ptr + row_id * y_stride + idx, mask=mask, other=0.0, cache_modifier=".ca")
        dx_blk = y_blk * (dy_blk - acc)
        tl.store(dx_ptr + row_id * dx_stride + idx, dx_blk, mask=mask, cache_modifier=".wb")
