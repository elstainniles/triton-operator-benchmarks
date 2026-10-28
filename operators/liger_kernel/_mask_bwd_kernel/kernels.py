# Copyright 2024 LinkedIn Corporation. BSD-2-Clause; see ../LICENSE.
# Exact function/decorators extracted from the pinned upstream module.
# Loaded in that audited module namespace by opbench.liger; not a standalone import.
@triton.jit
def _mask_bwd_kernel(
    grad_in_ptr, out_ptr, stride_b, stride_m, stride_n, L, BLOCK: tl.constexpr, num_warps: tl.constexpr
):
    row_block = tl.program_id(0)
    col_block = tl.program_id(1)
    batch_id = tl.program_id(2)

    row_idx = row_block * BLOCK + tl.arange(0, BLOCK)
    col_idx = col_block * BLOCK + tl.arange(0, BLOCK)
    in_bounds = (row_idx[:, None] < L) & (col_idx[None, :] < L)

    base = grad_in_ptr + batch_id * stride_b
    offs = row_idx[:, None] * stride_m + col_idx[None, :] * stride_n
    grad_vals = tl.load(base + offs, mask=in_bounds, other=0.0, cache_modifier=".ca")

    future = col_idx[None, :] > row_idx[:, None]
    zero = tl.zeros(grad_vals.shape, dtype=grad_vals.dtype)
    out = tl.where(future, zero, grad_vals)

    tl.store(out_ptr + batch_id * stride_b + offs, out, mask=in_bounds, cache_modifier=".wb")
