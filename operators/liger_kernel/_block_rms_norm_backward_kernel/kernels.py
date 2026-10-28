# Copyright 2024 LinkedIn Corporation. BSD-2-Clause; see ../LICENSE.
# Exact function/decorators extracted from the pinned upstream module.
# Loaded in that audited module namespace by opbench.liger; not a standalone import.
@triton.jit
def _block_rms_norm_backward_kernel(
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
    casting_mode: tl.constexpr,
    elementwise_affine: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_ROW: tl.constexpr,
    COMPUTE_DW: tl.constexpr = True,
):
    """
    dx = (1 / RMS) * [dy * (w + offset - (1 / N) * (1 / RMS^2) * ((dy * (w + offset)) dot x) * x]. * means element-wise multiplication, whileas dot means dot product
    dw = sum(dy * (x / RMS)). summation over BxT dimension
    """

    pid = tl.program_id(0).cast(tl.int64)
    NUM_SMS = tl.num_programs(0)

    col_offsets = tl.arange(0, BLOCK_SIZE)
    col_mask = col_offsets < n_cols

    if elementwise_affine and COMPUTE_DW:
        dW_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    if elementwise_affine:
        W_row = tl.load(W_ptr + col_offsets, mask=col_mask, other=0.0)
        # Pin the fp64 scalar Inductor passes to fp32 (see _rms_norm_forward_kernel).
        W_row = W_row + offset.to(tl.float32)

    for start in range(pid * BLOCK_ROW, n_rows, NUM_SMS * BLOCK_ROW):
        row_idx = start + tl.arange(0, BLOCK_ROW)
        row_mask = row_idx < n_rows
        dY_row = tl.load(
            dY_ptr + row_idx[:, None] * dY_row_stride + col_offsets[None, :],
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
        )
        X_row = tl.load(
            X_ptr + row_idx[:, None] * X_row_stride + col_offsets[None, :],
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
        )

        # Get cached rms
        rstd_row = tl.load(RSTD_ptr + row_idx * RSTD_row_stride, row_mask)

        X_row = X_row.to(tl.float32)

        # Different bacward graphs for different casting modes
        if casting_mode == _CASTING_MODE_LLAMA:
            if elementwise_affine:
                m = (dY_row * W_row[None, :]).to(tl.float32)
            else:
                m = dY_row.to(tl.float32)

        elif casting_mode == _CASTING_MODE_GEMMA:
            dY_row = dY_row.to(tl.float32)
            if elementwise_affine:
                m = dY_row * W_row[None, :]
            else:
                m = dY_row
        else:
            if elementwise_affine:
                m = dY_row * W_row[None, :]
            else:
                m = dY_row

        dX_row = rstd_row[:, None] * m

        dX_row += (rstd_row[:, None]) * (
            -(1 / n_cols) * (rstd_row * rstd_row * tl.sum(m * X_row, axis=1))[:, None] * X_row
        )

        if elementwise_affine and COMPUTE_DW:
            if casting_mode == _CASTING_MODE_LLAMA:
                # TODO(tcc): use tl.sum(..., dtype=tl.float32) once we upgrade to triton>=3.3.0
                dW_row += tl.sum((dY_row * (X_row * rstd_row[:, None]).to(X_dtype)).to(tl.float32), 0)
            else:
                # here X_row is already in fp32 (see previous if block)
                dW_row += tl.sum(dY_row * (X_row * rstd_row[:, None]), 0)

        tl.store(
            dX_ptr + row_idx[:, None] * dX_row_stride + col_offsets[None, :],
            dX_row,
            mask=row_mask[:, None] & col_mask[None, :],
        )

    if elementwise_affine and COMPUTE_DW:
        tl.store(dW_ptr + pid * dW_row_stride + col_offsets, dW_row, mask=col_mask)
