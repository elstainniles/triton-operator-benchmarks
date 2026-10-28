# Copyright 2024 LinkedIn Corporation. BSD-2-Clause; see ../LICENSE.
# Exact function/decorators extracted from the pinned upstream module.
# Loaded in that audited module namespace by opbench.liger; not a standalone import.
@triton.jit
def _block_rms_norm_forward_kernel(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    W_ptr,
    W_row_stride,
    RSTD_ptr,
    RSTD_row_stride,
    n_rows,
    n_cols,
    eps,
    offset,
    casting_mode: tl.constexpr,  # constexpr so the `if` blocks can be optimized out
    elementwise_affine: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_ROW: tl.constexpr,
):
    """
    y_i = (x_i / (RMS)) * (offset + wi), RMS = sqrt(sum(x_i^2) / N)

    Reference:
    1. https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
    2. https://github.com/unslothai/unsloth/blob/fd753fed99ed5f10ef8a9b7139588d9de9ddecfb/unsloth/kernels/rms_layernorm.py#L22
    3. https://arxiv.org/pdf/1910.07467
    """

    row_idx = tl.program_id(0).to(tl.int64) * BLOCK_ROW + tl.arange(0, BLOCK_ROW)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    row_mask = row_idx < n_rows
    col_mask = col_offsets < n_cols

    X_row = tl.load(
        X_ptr + row_idx[:, None] * X_row_stride + col_offsets[None, :],
        mask=row_mask[:, None] & col_mask[None, :],
        other=0,
    )
    X_row_dtype = X_row.dtype
    if elementwise_affine:
        W_row = tl.load(W_ptr + col_offsets, mask=col_mask, other=0)

    # On Llama, only rstd is computed on fp32
    if casting_mode == _CASTING_MODE_LLAMA:
        X_row = X_row.to(tl.float32)

    # Gemma computes everything on fp32, and then casts back the output to the original dtype
    if casting_mode == _CASTING_MODE_GEMMA:
        if elementwise_affine:
            W_row = W_row.to(tl.float32)
        X_row = X_row.to(tl.float32)

    if casting_mode == _CASTING_MODE_NONE:
        eps = eps.to(X_row_dtype)
        offset = offset.to(X_row_dtype)
    else:
        # See _rms_norm_forward_kernel: pin fp64 scalars from Inductor to fp32.
        eps = eps.to(tl.float32)
        offset = offset.to(tl.float32)

    mean_square = tl.sum(X_row * X_row, axis=1) / n_cols
    rstd = rsqrt(mean_square + eps)

    # We can save time by caching rms with minimal memory overhead
    # because rms is much smaller compared to X_row, as rms is for each row.
    # However, on the computation side, it can save 4 operations (*, sum, /, sqrt).
    tl.store(RSTD_ptr + row_idx * RSTD_row_stride, rstd, row_mask)

    X_row = X_row * rstd[:, None]

    # On Llama, the multiplication with the weight is done on the original dtype
    if casting_mode == _CASTING_MODE_LLAMA:
        X_row = X_row.to(X_row_dtype)

    if elementwise_affine:
        Y_row = X_row * (offset + W_row)[None, :]
    else:
        Y_row = X_row

    if casting_mode == _CASTING_MODE_GEMMA:
        Y_row = Y_row.to(X_row_dtype)

    tl.store(
        Y_ptr + row_idx[:, None] * Y_row_stride + col_offsets[None, :],
        Y_row,
        mask=row_mask[:, None] & col_mask[None, :],
    )
