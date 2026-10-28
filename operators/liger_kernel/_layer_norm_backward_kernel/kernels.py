# Copyright 2024 LinkedIn Corporation. BSD-2-Clause; see ../LICENSE.
# Exact function/decorators extracted from the pinned upstream module.
# Loaded in that audited module namespace by opbench.liger; not a standalone import.
@triton.jit
def _layer_norm_backward_kernel(
    X_ptr,  # pointer to input, shape (n_rows, n_cols)
    stride_x,  # stride of each row in input
    W_ptr,  # pointer to weights, shape (n_cols,)
    Mean_ptr,  # pointer to mean, shape (n_rows,)
    stride_mean,  # stride of each row in mean
    RSTD_ptr,  # pointer to rstd, shape (n_rows,)
    stride_rstd,  # stride of each row in rstd
    DX_ptr,  # pointer to input grad, shape (n_rows, n_cols)
    stride_dx,  # stride of each row in input grad
    DW_ptr,  # pointer to weights grad, shape (n_cols,)
    stride_dw,  # stride of each row in weights grad
    DB_ptr,  # pointer to bias grad, shape (n_cols,)
    stride_db,  # stride of each row in bias grad
    DY_ptr,  # pointer to output grad, shape (n_rows, n_cols)
    stride_dy,  # stride of each row in output grad
    n_rows,
    n_cols,
    rows_per_program: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    COMPUTE_DW: tl.constexpr = True,
    COMPUTE_DB: tl.constexpr = True,
):
    """
    References:
    https://arxiv.org/abs/1607.06450
    https://github.com/karpathy/llm.c/blob/master/doc/layernorm/layernorm.md
    """
    row_block_id = tl.program_id(0).to(tl.int64)
    row_start = row_block_id * rows_per_program
    row_end = min((row_block_id + 1) * rows_per_program, n_rows)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    if COMPUTE_DW:
        dW_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    if COMPUTE_DB:
        db_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Pre-load weights once (same optimization as forward pass)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0)
    w_f32 = w.to(tl.float32)

    for row_idx in range(row_start, row_end):
        # Calculate pointers for this specific row
        row_X_ptr = X_ptr + row_idx * stride_x
        row_DX_ptr = DX_ptr + row_idx * stride_dx
        row_DY_ptr = DY_ptr + row_idx * stride_dy
        row_Mean_ptr = Mean_ptr + row_idx * stride_mean
        row_RSTD_ptr = RSTD_ptr + row_idx * stride_rstd

        # Load data for this row
        x = tl.load(row_X_ptr + cols, mask=mask, other=0.0)
        dy = tl.load(row_DY_ptr + cols, mask=mask, other=0.0)
        mean = tl.load(row_Mean_ptr)
        rstd = tl.load(row_RSTD_ptr)

        # Convert to fp32 for numerical stability
        x_f32 = x.to(tl.float32)
        dy_f32 = dy.to(tl.float32)
        mean_f32 = mean.to(tl.float32)
        rstd_f32 = rstd.to(tl.float32)

        # Compute backward pass for this row
        x_hat = (x_f32 - mean_f32) * rstd_f32
        wdy = w_f32 * dy_f32
        c1 = tl.sum(x_hat * wdy, axis=0) / n_cols
        c2 = tl.sum(wdy, axis=0) / n_cols
        dx = (wdy - (x_hat * c1 + c2)) * rstd_f32

        # Store input gradient
        tl.store(row_DX_ptr + cols, dx, mask=mask)

        # Accumulate weight and bias gradients for this thread block's assigned rows
        if COMPUTE_DW:
            dw = dy_f32 * x_hat
            dW_row += dw
        if COMPUTE_DB:
            db = dy_f32
            db_row += db

    if COMPUTE_DW:
        tl.store(DW_ptr + row_block_id * stride_dw + cols, dW_row, mask=mask)
    if COMPUTE_DB:
        tl.store(DB_ptr + row_block_id * stride_db + cols, db_row, mask=mask)
