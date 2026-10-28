# Copyright 2024 LinkedIn Corporation. BSD-2-Clause; see ../LICENSE.
# Exact function/decorators extracted from the pinned upstream module.
# Loaded in that audited module namespace by opbench.liger; not a standalone import.
@triton.jit
def _group_norm_forward_kernel(
    Y_ptr,  # pointer to output, shape (n_rows, n_groups, hidden_size)
    Y_row_stride,  # stride of each row in output
    Y_col_stride,  # stride of each column in output
    X_ptr,  # pointer to input, shape (n_rows, n_groups, hidden_size)
    X_row_stride,  # stride of each row in input
    X_col_stride,  # stride of each column in input
    Mean_ptr,  # pointer to mean, shape (n_rows, n_groups)
    Mean_row_stride,  # stride of each row in mean
    Mean_col_stride,  # stride of each column in mean
    RSTD_ptr,  # pointer to rstd, shape (n_rows, n_groups)
    RSTD_row_stride,  # stride of each row in rstd
    RSTD_col_stride,  # stride of each column in rstd
    W_ptr,  # pointer to W
    B_ptr,  # pointer to B
    hidden_size,  # hidden size of X
    channels_per_group,  # the number of channels per group
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    References:
    https://nn.labml.ai/normalization/group_norm/index.html
    """
    batch_idx = tl.program_id(0)
    group_idx = tl.program_id(1)

    X_ptr += batch_idx * X_row_stride + group_idx * X_col_stride
    Y_ptr += batch_idx * Y_row_stride + group_idx * Y_col_stride

    block_range = tl.arange(0, BLOCK_SIZE)

    # Compute mean and variance using the online algorithm
    s = 0.0
    squared_sum = 0.0
    for i in tl.range(0, hidden_size, BLOCK_SIZE):
        hidden_size_offsets = i + block_range
        mask = hidden_size_offsets < hidden_size
        # upcast before the sums: fp16 sum(X * X) overflows and bf16 E[X^2] - E[X]^2 cancels
        X = tl.load(X_ptr + hidden_size_offsets, mask=mask, other=0.0).to(tl.float32)
        s += tl.sum(X)
        # X**2
        squared_sum += tl.sum(X * X)

    m = s / hidden_size

    # variance = E[X**2] - E[X]**2
    variance = (squared_sum / hidden_size) - (m * m)

    # 1/std
    rstd = rsqrt(variance + eps.to(tl.float32))

    # Normalize — flat loop over full hidden_size (not per-channel)
    # This avoids the nested channel × per_channel_hidden loop where
    # BLOCK_SIZE >> hidden_size_per_channel causes massive padding waste.
    hidden_size_per_channel = hidden_size // channels_per_group
    for i in tl.range(0, hidden_size, BLOCK_SIZE):
        hidden_size_offsets = i + block_range
        mask = hidden_size_offsets < hidden_size
        X = tl.load(X_ptr + hidden_size_offsets, mask=mask, other=m)
        # Determine which channel each element belongs to, then load W/B
        local_channel = hidden_size_offsets // hidden_size_per_channel
        global_channel = group_idx * channels_per_group + local_channel
        W = tl.load(W_ptr + global_channel, mask=mask)
        B = tl.load(B_ptr + global_channel, mask=mask)
        Y = (X - m) * rstd * W + B
        tl.store(Y_ptr + hidden_size_offsets, Y, mask=mask)

    tl.store(Mean_ptr + batch_idx * Mean_row_stride + group_idx * Mean_col_stride, m)
    tl.store(RSTD_ptr + batch_idx * RSTD_row_stride + group_idx * RSTD_col_stride, rstd)
