# Copyright 2024 LinkedIn Corporation. BSD-2-Clause; see ../LICENSE.
# Exact function/decorators extracted from the pinned upstream module.
# Loaded in that audited module namespace by opbench.liger; not a standalone import.
@triton.jit
def _poly_norm_forward_kernel(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    W_ptr,  # weight: [3] for [w0, w1, w2]
    B_ptr,  # bias: scalar
    RSTD_ptr,  # cache rstd for backward: shape (n_rows, 3)
    RSTD_row_stride,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    PolyNorm formula:
        y = w₀·norm(x³) + w₁·norm(x²) + w₂·norm(x) + b
        where norm(u) = u / sqrt(mean(u²) + ε)

    Reference:
    1. https://github.com/BryceZhuo/PolyCom/
    2. https://arxiv.org/pdf/2411.03884

    Cache rstd values for backward pass
    """
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load pointers
    Y_ptr += row_idx * Y_row_stride
    X_ptr += row_idx * X_row_stride
    RSTD_ptr += row_idx * RSTD_row_stride

    # Load input row; compute in fp32 (sum(x^6) overflows fp16 at hidden size ~4096)
    X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

    # Load weights and bias
    w0 = tl.load(W_ptr + 0)
    w1 = tl.load(W_ptr + 1)
    w2 = tl.load(W_ptr + 2)
    b = tl.load(B_ptr)

    # Compute x³, x², x
    X_pow3 = X_row * X_row * X_row
    X_pow2 = X_row * X_row
    X_pow1 = X_row

    # Compute norm(x³): norm(u) = u * rsqrt(mean(u²) + eps)
    # Force fp32 eps: Inductor may use fp64 under torch.compile and tank throughput.
    eps = eps.to(tl.float32)

    mean_square_3 = tl.sum(X_pow3 * X_pow3, axis=0) / n_cols
    rstd_3 = rsqrt(mean_square_3 + eps)
    norm_x3 = X_pow3 * rstd_3

    # Compute norm(x²)
    mean_square_2 = tl.sum(X_pow2 * X_pow2, axis=0) / n_cols
    rstd_2 = rsqrt(mean_square_2 + eps)
    norm_x2 = X_pow2 * rstd_2

    # Compute norm(x)
    mean_square_1 = tl.sum(X_pow1 * X_pow1, axis=0) / n_cols
    rstd_1 = rsqrt(mean_square_1 + eps)
    norm_x1 = X_pow1 * rstd_1

    # Cache rstd values for backward
    tl.store(RSTD_ptr + 0, rstd_3)
    tl.store(RSTD_ptr + 1, rstd_2)
    tl.store(RSTD_ptr + 2, rstd_1)

    # Compute output: y = w₀·norm(x³) + w₁·norm(x²) + w₂·norm(x) + b
    Y_row = w0 * norm_x3 + w1 * norm_x2 + w2 * norm_x1 + b

    # Store output
    tl.store(Y_ptr + col_offsets, Y_row, mask=mask)
