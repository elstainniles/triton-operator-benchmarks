# Copyright 2024 LinkedIn Corporation. BSD-2-Clause; see ../LICENSE.
# Exact function/decorators extracted from the pinned upstream module.
# Loaded in that audited module namespace by opbench.liger; not a standalone import.
@triton.jit
def _mask_fwd_kernel(
    scores_ptr,
    out_ptr,
    stride_b,
    stride_m,
    stride_n,
    L,
    mask_val: tl.constexpr,
    BLOCK: tl.constexpr,
    num_warps: tl.constexpr,
):
    row_block = tl.program_id(0)
    col_block = tl.program_id(1)
    batch_id = tl.program_id(2)

    row_idx = row_block * BLOCK + tl.arange(0, BLOCK)
    col_idx = col_block * BLOCK + tl.arange(0, BLOCK)
    in_bounds = (row_idx[:, None] < L) & (col_idx[None, :] < L)

    base = scores_ptr + batch_id * stride_b
    offs = row_idx[:, None] * stride_m + col_idx[None, :] * stride_n
    future = col_idx[None, :] > row_idx[:, None]
    mask_load = in_bounds & ~future
    out = tl.load(base + offs, mask=mask_load, other=mask_val, cache_modifier=".ca")
    tl.store(out_ptr + batch_id * stride_b + offs, out, mask=in_bounds, cache_modifier=".cs")
