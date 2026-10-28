# Copyright 2024 LinkedIn Corporation. BSD-2-Clause; see ../LICENSE.
# Exact function/decorators extracted from the pinned upstream module.
# Loaded in that audited module namespace by opbench.liger; not a standalone import.
@triton.jit
def _fused_add_rms_norm_backward_kernel(
    dY_ptr,
    dY_row_stride,
    dS_out_ptr,
    dS_out_row_stride,
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
    rows_per_program: tl.constexpr,
    casting_mode: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    has_dS_out: tl.constexpr,
    COMPUTE_DW: tl.constexpr = True,
):
    """
    This kernel is adapted from the rms_norm backward kernel, and is adapted to support the residual
    addition in the backward pass. For the following code pattern:
    1. hidden_states = residual + hidden_states
    2. residual = hidden_states
    3. hidden_states = rmsnorm(hidden_states)

    The gradient of hidden_states and residual comes out be exactly same. The value of this gradient is
    the sum of the gradient of the hidden_states in step 3 and the gradient of the residual in step 2.

    The backward pass computation logic is same as the rms_norm backward kernel, except that the gradient
    of the hidden_states in step 3 and the gradient of the residual in step 2 are summed up.

    NOTE: The computation order below is intentionally structured to reduce register pressure.
    dW is accumulated before computing dX so the compiler can reuse dY_row's registers for m.
    The dX formula is factored as `rstd * m + c * X` (c is a scalar) to minimize live vectors.
    """

    row_block_id = tl.program_id(0).to(tl.int64)
    row_start = row_block_id * rows_per_program
    row_end = min((row_block_id + 1) * rows_per_program, n_rows)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    if COMPUTE_DW:
        dW_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)
    W_row = W_row + offset.to(tl.float32)

    for row_idx in range(row_start, row_end):
        dy_base = dY_ptr + row_idx * dY_row_stride
        dx_base = dX_ptr + row_idx * dX_row_stride
        x_base = X_ptr + row_idx * X_row_stride
        rstd_base = RSTD_ptr + row_idx * RSTD_row_stride

        dY_row = tl.load(dy_base + col_offsets, mask=mask, other=0.0)
        X_row = tl.load(x_base + col_offsets, mask=mask, other=0.0)
        rstd_row = tl.load(rstd_base)

        X_row = X_row.to(tl.float32)

        # Calculate the gradient of W first to reduce peak register liveness
        if COMPUTE_DW:
            if casting_mode == _CASTING_MODE_LLAMA:
                dW_row += dY_row * (X_row * rstd_row).to(X_dtype)
            else:
                # here X_row is already in fp32 (see previous cast)
                dW_row += dY_row * (X_row * rstd_row)

        # Different backward graphs for different casting modes
        if casting_mode == _CASTING_MODE_LLAMA:
            m = (dY_row * W_row).to(tl.float32)
        elif casting_mode == _CASTING_MODE_GEMMA:
            dY_row = dY_row.to(tl.float32)
            m = dY_row * W_row
        else:
            m = dY_row * W_row

        # dX = rstd * m - (rstd^3 / n_cols) * dot(m, X) * X + dS_out
        dot = tl.sum(m * X_row, axis=0)
        c = -(1.0 / n_cols) * rstd_row * rstd_row * rstd_row * dot
        dX_row = rstd_row * m + c * X_row

        if has_dS_out:
            ds_base = dS_out_ptr + row_idx * dS_out_row_stride
            dS_out_row = tl.load(ds_base + col_offsets, mask=mask, other=0.0)
            dX_row += dS_out_row

        tl.store(dx_base + col_offsets, dX_row.to(X_dtype), mask=mask)

    if COMPUTE_DW:
        tl.store(dW_ptr + row_block_id * dW_row_stride + col_offsets, dW_row, mask=mask)
