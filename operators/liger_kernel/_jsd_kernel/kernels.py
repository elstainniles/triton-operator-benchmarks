# Copyright 2024 LinkedIn Corporation. BSD-2-Clause; see ../LICENSE.
# Exact function/decorators extracted from the pinned upstream module.
# Loaded in that audited module namespace by opbench.liger; not a standalone import.
@triton.jit
def _jsd_kernel(
    X_ptr,  # input in logspace, X = log Q
    X_stride,
    Y_ptr,  # ground truth in logspace, Y = log P
    Y_stride,
    loss_ptr,
    loss_stride,
    dX_ptr,
    dX_stride,
    label_ptr,
    beta: tl.constexpr,
    n_non_ignore: int,
    ignore_index: tl.constexpr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    HAS_LABEL: tl.constexpr,
):
    # JSD(P || Q) = (KL(P || M) + KL(Q || M)) / 2, M = (1/2) * (P + Q) = (1/2) * (e ^ Y + e ^ X)
    #             = sum(P * log P + Q * log Q - 2 * M * log M) / 2
    #             = sum(e ^ Y * Y + e ^ X * X - 2 * M * log M) / 2
    # grad_x_i = 0.5 * Q * (X - log_M)
    pid = tl.program_id(0).to(tl.int64)
    X_ptr += pid * X_stride
    dX_ptr += pid * dX_stride
    Y_ptr += pid * Y_stride
    loss_ptr += pid * loss_stride
    label_ptr += pid

    if HAS_LABEL:
        label = tl.load(label_ptr)
        if label == ignore_index:
            for i in range(0, n_cols, BLOCK_SIZE):
                offsets = i + tl.arange(0, BLOCK_SIZE)
                tl.store(dX_ptr + offsets, 0.0, mask=offsets < n_cols)
            return

    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        X = tl.load(X_ptr + offsets, mask=mask, other=float("-inf")).to(tl.float32)
        Y = tl.load(Y_ptr + offsets, mask=mask, other=float("-inf")).to(tl.float32)

        if beta == 0.0:  # forward KL
            Y_max = tl.max(Y, axis=0)
            Y_shifted = Y - Y_max
            Y_prob = tl.exp(Y_shifted) * tl.exp(Y_max)  # Compensate for the shift
            loss = Y_prob * (Y - X)
            dX = -Y_prob
        elif beta == 1.0:  # reverse KL
            X_max = tl.max(X, axis=0)
            X_shifted = X - X_max
            X_prob = tl.exp(X_shifted) * tl.exp(X_max)  # Compensate for the shift
            loss = X_prob * (X - Y)
            dX = loss + X_prob
        else:
            max_val = tl.maximum(X, Y)
            X_shifted = X - max_val
            Y_shifted = Y - max_val

            # Compute log(M) before rescaling to probability space. Computing M
            # directly can underflow to zero for low-probability vocabulary blocks.
            M_shifted = beta * tl.exp(Y_shifted) + (1 - beta) * tl.exp(X_shifted)
            log_M = max_val + tl.log(M_shifted)

            Q = tl.exp(X)
            P = tl.exp(Y)
            beta_P = beta * P
            one_minus_beta_Q = (1 - beta) * Q

            loss = beta_P * (Y - log_M) + one_minus_beta_Q * (X - log_M)
            dX = one_minus_beta_Q * (X - log_M)

        # Pre-compute scaling factor
        scale = 1.0 / n_non_ignore
        loss = loss * scale
        dX = dX * scale

        tl.store(loss_ptr + offsets, loss, mask=mask)
        tl.store(dX_ptr + offsets, dX, mask=mask)
