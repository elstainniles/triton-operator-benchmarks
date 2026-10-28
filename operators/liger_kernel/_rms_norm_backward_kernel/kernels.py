# Copyright 2024 LinkedIn Corporation. BSD-2-Clause; see ../LICENSE.
# Exact function/decorators extracted from the pinned upstream module.
# Loaded in that audited module namespace by opbench.liger; not a standalone import.
@triton.jit
def _rms_norm_backward_kernel(
    dY_ptr,
    dY_row_stride,
    dX_ptr,
    dX_row_stride,
    X_ptr,
    X_row_stride,
    X_dtype: tl.constexpr,
    W_ptr,
    W_row_stride,
    RSTD_ptr,
    RSTD_row_stride,
    dW_ptr,
    dW_row_stride,
    n_rows,
    n_cols,
    offset,
    rows_per_program,
    casting_mode: tl.constexpr,
    elementwise_affine: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    COMPUTE_DW: tl.constexpr = True,
):
    """
    dx = (1 / RMS) * [dy * (w + offset - (1 / N) * (1 / RMS^2) * ((dy * (w + offset)) dot x) * x]. * means element-wise multiplication, whileas dot means dot product
    dw = sum(dy * (x / RMS)). summation over BxT dimension
    """

    row_block_id = tl.program_id(0).to(tl.int64)
    row_start = row_block_id * rows_per_program
    row_end = min((row_block_id + 1) * rows_per_program, n_rows)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    if elementwise_affine and COMPUTE_DW:
        dW_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    if elementwise_affine:
        W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)
        # Pin the fp64 scalar Inductor passes to fp32 (see _rms_norm_forward_kernel).
        W_row = W_row + offset.to(tl.float32)

    for row_idx in range(row_start, row_end):
        dy_base = dY_ptr + row_idx * dY_row_stride
        dx_base = dX_ptr + row_idx * dX_row_stride

        x_base = X_ptr + row_idx * X_row_stride
        rstd_base = RSTD_ptr + row_idx * RSTD_row_stride

        dY_row = tl.load(dy_base + col_offsets, mask=mask, other=0.0)
        X_row = tl.load(x_base + col_offsets, mask=mask, other=0.0)

        # Get cached rms
        rstd_row = tl.load(rstd_base)

        X_row = X_row.to(tl.float32)

        # Different backward graphs for different casting modes
        if casting_mode == _CASTING_MODE_LLAMA:
            if elementwise_affine:
                m = (dY_row * W_row).to(tl.float32)
            else:
                m = dY_row.to(tl.float32)

        elif casting_mode == _CASTING_MODE_GEMMA:
            dY_row = dY_row.to(tl.float32)
            if elementwise_affine:
                m = dY_row * W_row
            else:
                m = dY_row
        else:
            if elementwise_affine:
                m = dY_row * W_row
            else:
                m = dY_row

        dX_row = rstd_row * m

        dX_row += (rstd_row) * (-(1 / n_cols) * rstd_row * rstd_row * tl.sum(m * X_row, axis=0) * X_row)

        if elementwise_affine and COMPUTE_DW:
            # calculate the gradient of W
            if casting_mode == _CASTING_MODE_LLAMA:
                dW_row += dY_row * (X_row * rstd_row).to(X_dtype)
            else:
                # here X_row is already in fp32 (see previous if block)
                dW_row += dY_row * (X_row * rstd_row)

        tl.store(dx_base + col_offsets, dX_row.to(X_dtype), mask=mask)

    if elementwise_affine and COMPUTE_DW:
        tl.store(dW_ptr + row_block_id * dW_row_stride + col_offsets, dW_row, mask=mask)
