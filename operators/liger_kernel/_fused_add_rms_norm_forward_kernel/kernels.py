# Copyright 2024 LinkedIn Corporation. BSD-2-Clause; see ../LICENSE.
# Exact function/decorators extracted from the pinned upstream module.
# Loaded in that audited module namespace by opbench.liger; not a standalone import.
@triton.jit
def _fused_add_rms_norm_forward_kernel(
    Y_ptr,
    Y_row_stride,
    S_ptr,  # output residual
    S_row_stride,
    X_ptr,
    X_row_stride,
    R_ptr,  # input residual
    R_row_stride,
    W_ptr,
    W_row_stride,
    RSTD_ptr,
    RSTD_row_stride,
    n_cols,
    eps,
    offset,
    casting_mode: tl.constexpr,  # constexpr so the `if` blocks can be optimized out
    BLOCK_SIZE: tl.constexpr,
):
    """
    This kernel computes the following:
    1. hidden_states = residual + hidden_states
    2. residual = hidden_states
    3. hidden_states = rmsnorm(hidden_states)

    This is a commonly used pattern in the decoder layers of LLMs.
    Some examples:
    1. https://github.com/huggingface/transformers/blob/0dc2df5ddafe3cb5824ad24e85beba13e0aa6726/src/transformers/models/qwen3/modeling_qwen3.py#L271
    2. https://github.com/huggingface/transformers/blob/0dc2df5ddafe3cb5824ad24e85beba13e0aa6726/src/transformers/models/llama4/modeling_llama4.py#L393

    This kernel is inspired by the rms_norm forward kernel, and is adapted to support the residual addition in the forward pass.
    The backward pass is also adapted to support the residual addition in the backward pass.
    """

    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y_ptr += row_idx * Y_row_stride
    S_ptr += row_idx * S_row_stride
    X_ptr += row_idx * X_row_stride
    R_ptr += row_idx * R_row_stride
    RSTD_ptr += row_idx * RSTD_row_stride

    X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0)
    R_row = tl.load(R_ptr + col_offsets, mask=mask, other=0)
    S_row = X_row + R_row
    tl.store(S_ptr + col_offsets, S_row, mask=mask)
    S_row_dtype = S_row.dtype
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0)

    # On Llama, only rstd is computed on fp32
    if casting_mode == _CASTING_MODE_LLAMA:
        S_row = S_row.to(tl.float32)

    # Gemma computes everything on fp32, and then casts back the output to the original dtype
    if casting_mode == _CASTING_MODE_GEMMA:
        W_row = W_row.to(tl.float32)
        S_row = S_row.to(tl.float32)

    if casting_mode == _CASTING_MODE_NONE:
        eps = eps.to(S_row_dtype)
        offset = offset.to(S_row_dtype)
    else:
        # Force fp32 scalars: Inductor may use fp64 under torch.compile and tank throughput.
        eps = eps.to(tl.float32)
        offset = offset.to(tl.float32)

    mean_square = tl.sum(S_row * S_row, axis=0) / n_cols
    rstd = rsqrt(mean_square + eps)

    # We can save time by caching rms with minimal memory overhead
    # because rms is much smaller compared to X_row, as rms is for each row.
    # However, on the computation side, it can save 4 operations (*, sum, /, sqrt).
    tl.store(RSTD_ptr, rstd)

    S_row = S_row * rstd

    # On Llama, the multiplication with the weight is done on the original dtype
    if casting_mode == _CASTING_MODE_LLAMA:
        S_row = S_row.to(S_row_dtype)

    Y_row = S_row * (offset + W_row)

    if casting_mode == _CASTING_MODE_GEMMA:
        Y_row = Y_row.to(S_row_dtype)

    tl.store(Y_ptr + col_offsets, Y_row, mask=mask)
