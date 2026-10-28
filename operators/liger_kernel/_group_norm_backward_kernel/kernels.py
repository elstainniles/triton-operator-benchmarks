# Copyright 2024 LinkedIn Corporation. BSD-2-Clause; see ../LICENSE.
# Exact function/decorators extracted from the pinned upstream module.
# Loaded in that audited module namespace by opbench.liger; not a standalone import.
@triton.jit
def _group_norm_backward_kernel(
    X_ptr,  # pointer to input, shape (n_rows, n_channels, hidden_size)
    X_row_stride,  # stride of each row in input
    X_col_stride,  # stride of each column in input
    W_ptr,  # pointer to weights, shape (n_channels)
    Mean_ptr,  # pointer to mean, shape (n_rows, n_groups)
    Mean_ptr_row_stride,  # stride of each column in mean
    Mean_ptr_col_stride,  # stride of each column in mean
    RSTD_ptr,  # pointer to rstd, shape (n_rows, n_groups)
    DX_ptr,  # pointer to input grad, shape (n_rows, n_groups, hidden_size)
    DW_ptr,  # pointer to weights grad, shape (n_channels)
    DB_ptr,  # pointer to bias grad, shape (n_channels)
    UPSTREAM_ptr,  # pointer to output grad, shape (n_rows, n_channels, hidden_size)
    hidden_size: tl.constexpr,  # hidden size
    channels_per_group: tl.constexpr,  # number of groups in group norm
    BLOCK_SIZE: tl.constexpr,
    dtype: tl.constexpr,
):
    """
    References:
    https://nn.labml.ai/normalization/group_norm/index.html
    https://github.com/karpathy/llm.c/blob/master/doc/layernorm/layernorm.md

    The backprop equations are the same for group_norm and layer_norm
    the only difference here is that we load the Mean, Rstd corresponding to the
    group we're computing gradients for and the mean and rstd are computed over n-channels
    so the total number of elements we compute the mean over is num_channels_per_group * hidden_size

    We also need to load the Weights corresponding to the current channel to compute the gradients.
    """
    batch_idx = tl.program_id(0)
    group_idx = tl.program_id(1)

    # Move the pointers to the correct batch
    X_ptr += batch_idx * X_row_stride
    DX_ptr += batch_idx * X_row_stride
    UPSTREAM_ptr += batch_idx * X_row_stride

    # Mean and rstd are the same shape so have the same strides
    mean = tl.load(Mean_ptr + batch_idx * Mean_ptr_row_stride + group_idx * Mean_ptr_col_stride)
    rstd = tl.load(RSTD_ptr + batch_idx * Mean_ptr_row_stride + group_idx * Mean_ptr_col_stride)

    c1 = 0.0
    c2 = 0.0
    block_range = tl.arange(0, BLOCK_SIZE)

    # We need to compute the sum terms of the backprop equations across all channels in the group
    for channel_idx in range(group_idx * channels_per_group, (group_idx + 1) * channels_per_group):
        dW = 0.0
        dB = 0.0
        # Move the pointers to the correct channel
        W = tl.load(W_ptr + channel_idx)
        for i in tl.range(0, hidden_size, BLOCK_SIZE):
            hidden_size_offsets = i + block_range
            mask = hidden_size_offsets < hidden_size
            X = tl.load(
                X_ptr + channel_idx * X_col_stride + hidden_size_offsets,
                mask=mask,
                other=0.0,
            )
            UPSTREAM_grad = tl.load(
                UPSTREAM_ptr + channel_idx * X_col_stride + hidden_size_offsets,
                mask=mask,
                other=0.0,
            )

            x_hat = (X - mean) * rstd
            dW += tl.sum(UPSTREAM_grad * x_hat)
            dB += tl.sum(UPSTREAM_grad)

            wdy = W * UPSTREAM_grad
            c1 += tl.sum(x_hat * wdy)
            c2 += tl.sum(wdy)

        # Need to ensure additions to the same channel are atomic
        tl.atomic_add(DW_ptr + channel_idx, dW.to(dtype))
        tl.atomic_add(DB_ptr + channel_idx, dB.to(dtype))

    N = hidden_size * channels_per_group
    c1 = c1 / N
    c2 = c2 / N

    for channel_idx in tl.range(group_idx * channels_per_group, (group_idx + 1) * channels_per_group):
        # Move the pointers to the correct channel
        W = tl.load(W_ptr + channel_idx)
        for i in range(0, hidden_size, BLOCK_SIZE):
            hidden_size_offsets = i + block_range
            mask = hidden_size_offsets < hidden_size
            X = tl.load(
                X_ptr + channel_idx * X_col_stride + hidden_size_offsets,
                mask=mask,
                other=0.0,
            )
            UPSTREAM_grad = tl.load(
                UPSTREAM_ptr + channel_idx * X_col_stride + hidden_size_offsets,
                mask=mask,
                other=0.0,
            )

            x_hat = (X - mean) * rstd
            wdy = W * UPSTREAM_grad
            dx = (wdy - (x_hat * c1 + c2)) * rstd
            tl.store(DX_ptr + channel_idx * X_col_stride + hidden_size_offsets, dx, mask=mask)
