# Copyright 2024 LinkedIn Corporation. BSD-2-Clause; see ../LICENSE.
# Exact function/decorators extracted from the pinned upstream module.
# Loaded in that audited module namespace by opbench.liger; not a standalone import.
@triton.jit
def _sparsemax_forward_kernel(
    x_ptr,
    x_stride_row,
    sorted_x_ptr,
    sorted_x_stride_row,
    o_ptr,
    o_stride_row,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    num_warps: tl.constexpr,
):
    pid_row = tl.program_id(0).to(tl.int64)
    ptr_x_data_row = x_ptr + pid_row * x_stride_row
    ptr_sorted_x_data_row = sorted_x_ptr + pid_row * sorted_x_stride_row
    ptr_output_row = o_ptr + pid_row * o_stride_row

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols

    z_sorted_block = tl.load(
        ptr_sorted_x_data_row + offs,
        mask=mask,
        other=-float("inf"),
        cache_modifier=".cg",
    ).to(tl.float32)

    # sparsemax is shift-invariant: subtract the row max so that a large common offset does not
    # absorb the "- 1" in the threshold (rows whose max is not finite are left unshifted)
    z_max = tl.load(ptr_sorted_x_data_row).to(tl.float32)
    z_max = tl.where(tl.abs(z_max) < float("inf"), z_max, 0.0)
    z_sorted_block = z_sorted_block - z_max

    z_valid = tl.where(mask, z_sorted_block, 0.0)
    cssv = tl.cumsum(z_valid, 0)

    r = (offs + 1).to(tl.float32)
    t_vec = (cssv - 1.0) / r

    support = (z_sorted_block > t_vec) & mask

    k_int = tl.sum(support.to(tl.int32), 0)
    k_clamped_int = tl.maximum(k_int, 1)
    k = k_clamped_int.to(tl.float32)

    s = tl.sum(tl.where(support, z_sorted_block, 0.0), 0)

    tau = (s - 1.0) / k

    x_block = tl.load(
        ptr_x_data_row + offs,
        mask=mask,
        other=0.0,
        cache_modifier=".cg",
    ).to(tl.float32)

    y = tl.maximum((x_block - z_max) - tau, 0.0)

    tl.store(
        ptr_output_row + offs,
        y.to(ptr_output_row.dtype.element_ty),
        mask=mask,
        cache_modifier=".cs",
    )
