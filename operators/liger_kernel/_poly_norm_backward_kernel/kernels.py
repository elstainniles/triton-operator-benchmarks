# Copyright 2024 LinkedIn Corporation. BSD-2-Clause; see ../LICENSE.
# Exact function/decorators extracted from the pinned upstream module.
# Loaded in that audited module namespace by opbench.liger; not a standalone import.
@triton.jit
def _poly_norm_backward_kernel(
    dY_ptr,
    dY_row_stride,
    dX_ptr,
    dX_row_stride,
    X_ptr,
    X_row_stride,
    W_ptr,
    RSTD_ptr,
    RSTD_row_stride,
    dW_ptr,  # shape: (n_programs, 3)
    dW_row_stride,
    dB_ptr,  # shape: (n_programs,)
    n_rows,
    n_cols,
    rows_per_program: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    PolyNorm Backward Kernel Gradient:
        ∂L/∂x_i = Σ_p w_p * [p*x_i^(p-1) * grad_i/D_p - (p/d)*x_i^(2p-1) * S_p/(D_p³)]

    where:
        - D_p = RMS(x^p) = 1/rstd_p
        - S_p = sum(grad * x^p) over the row
        - d = n_cols
        - p ∈ {3, 2, 1}
    """
    row_block_id = tl.program_id(0).to(tl.int64)
    row_start = row_block_id * rows_per_program
    row_end = min((row_block_id + 1) * rows_per_program, n_rows)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Initialize accumulators for weight and bias gradients (scalars)
    dW0_acc = 0.0
    dW1_acc = 0.0
    dW2_acc = 0.0
    dB_acc = 0.0

    # Load weights
    w0 = tl.load(W_ptr + 0).to(tl.float32)
    w1 = tl.load(W_ptr + 1).to(tl.float32)
    w2 = tl.load(W_ptr + 2).to(tl.float32)

    for row_idx in range(row_start, row_end):
        dy_base = dY_ptr + row_idx * dY_row_stride
        x_base = X_ptr + row_idx * X_row_stride
        dx_base = dX_ptr + row_idx * dX_row_stride
        rstd_base = RSTD_ptr + row_idx * RSTD_row_stride

        dY_row = tl.load(dy_base + col_offsets, mask=mask, other=0.0).to(tl.float32)
        X_row = tl.load(x_base + col_offsets, mask=mask, other=0.0).to(tl.float32)

        # Load cached rstd values
        rstd_3 = tl.load(rstd_base + 0).to(tl.float32)
        rstd_2 = tl.load(rstd_base + 1).to(tl.float32)
        rstd_1 = tl.load(rstd_base + 2).to(tl.float32)

        # Compute powers
        X_pow3 = X_row * X_row * X_row
        X_pow2 = X_row * X_row
        X_pow1 = X_row

        # Accumulate bias gradient: dB = sum(dY)
        dB_acc += tl.sum(dY_row, axis=0)

        # Compute gradient w.r.t. input using closed-form formula
        # For p=3: ∂L/∂x from w0 * norm(x³)
        S_3 = tl.sum(dY_row * X_pow3, axis=0)  # scalar
        grad_x_3 = w0 * (
            3.0 * X_pow2 * rstd_3 * dY_row
            - (3.0 / n_cols) * X_row * X_row * X_row * X_row * X_row * (rstd_3 * rstd_3 * rstd_3) * S_3
        )

        # For p=2: ∂L/∂x from w1 * norm(x²)
        S_2 = tl.sum(dY_row * X_pow2, axis=0)  # scalar
        grad_x_2 = w1 * (
            2.0 * X_row * rstd_2 * dY_row - (2.0 / n_cols) * X_row * X_row * X_row * (rstd_2 * rstd_2 * rstd_2) * S_2
        )

        # For p=1: ∂L/∂x from w2 * norm(x)
        S_1 = tl.sum(dY_row * X_pow1, axis=0)  # scalar
        grad_x_1 = w2 * (1.0 * rstd_1 * dY_row - (1.0 / n_cols) * X_row * (rstd_1 * rstd_1 * rstd_1) * S_1)

        # Accumulate weight gradients using closed-form: dW_p = rstd_p * S_p
        dW0_acc += rstd_3 * S_3
        dW1_acc += rstd_2 * S_2
        dW2_acc += rstd_1 * S_1

        # Total gradient
        dX_row = grad_x_3 + grad_x_2 + grad_x_1

        # Store gradient
        tl.store(dx_base + col_offsets, dX_row, mask=mask)

    # Store accumulated gradients (scalars)
    tl.store(dW_ptr + row_block_id * dW_row_stride + 0, dW0_acc)
    tl.store(dW_ptr + row_block_id * dW_row_stride + 1, dW1_acc)
    tl.store(dW_ptr + row_block_id * dW_row_stride + 2, dW2_acc)
    tl.store(dB_ptr + row_block_id, dB_acc)
