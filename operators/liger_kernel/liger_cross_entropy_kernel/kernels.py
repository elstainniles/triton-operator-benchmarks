# Copyright 2024 LinkedIn Corporation. BSD-2-Clause; see ../LICENSE.
# Exact function/decorators extracted from the pinned upstream module.
# Loaded in that audited module namespace by opbench.liger; not a standalone import.
@triton.jit
def liger_cross_entropy_kernel(
    X_ptr,
    X_stride,
    Y_ptr,
    Y_stride,
    weight_ptr,
    loss_ptr,
    z_loss_ptr,
    loss_stride,
    token_accuracy_ptr,
    token_accuracy_stride,
    predicted_tokens_ptr,
    predicted_tokens_stride,
    n_cols,
    n_non_ignore,
    sum_non_ignore_weight,
    weight_sum,
    ignore_index,
    lse_square_scale: tl.constexpr,
    label_smoothing: tl.constexpr,
    reduction: tl.constexpr,  # set it as constexpr since reduction is always known at compile time
    softcap,
    RETURN_Z_LOSS: tl.constexpr,
    RETURN_TOKEN_ACCURACY: tl.constexpr,
    RETURN_PREDICTED_TOKENS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_SOFTCAPPING: tl.constexpr,
    HAS_GRADIENTS: tl.constexpr,
):
    """
    This kernel computes both cross entropy loss and the gradient of the input.
    We only consider hard label + mean reduction for now. Please refer to https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html for the math.

    Parameters:
    X_ptr: Pointer to input tensor.
    X_stride (int): The stride of the input tensor.
    Y_ptr: Pointer to target tensor.
    Y_stride (int): The stride of the target tensor.
    weight_ptr: Pointer to weight tensor.
    loss_ptr: Pointer to tensor to store the loss.
    z_loss_ptr: Pointer to tensor to store the z loss. No operation if RETURN_Z_LOSS is 0.
    loss_stride (int): The stride of the loss tensor.
    token_accuracy_ptr: Pointer to tensor to store the per-token accuracy. No operation if RETURN_TOKEN_ACCURACY is 0.
    token_accuracy_stride (int): The stride of the token accuracy tensor.
    n_cols (int): The number of columns in the input tensor.
    n_non_ignore (float): The number of non-ignored elements in the batch.
    sum_non_ignore_weight (float): The sum of non-ignored target's weights in the batch.
    weight_sum (float): The sum of weight tensor.
    ignore_index (int): The index to ignore in the target.
    label_smoothing (float): The amount of smoothing when computing the loss, where 0.0 means no smoothing.
    lse_square_scale (float): The scaler of (logsumexp(_input)) ^ 2 adding to the loss for the stability of training.
    reduction (str): The string for the reduction to apply
    softcap (float): The upper threshold for scaling logits to the range (-softcap, +softcap).
    RETURN_Z_LOSS (int): The boolean value to decide whether to store z loss to z_loss_ptr or not. It must be 0 or 1.
    RETURN_TOKEN_ACCURACY (int): The boolean value to decide whether to store per-token accuracy to token_accuracy_ptr or not. It must be 0 or 1.
    BLOCK_SIZE (int): The block size for Triton operations.
    HAS_WEIGHT (bool): The boolean value to determine whether assigning weight to each of the classes.
    HAS_SOFTCAPPING (bool): The boolean value to determine whether applying soft-capping or not.
    HAS_GRADIENTS (bool): The boolean value to determine whether calculating gradients in forward pass.
    """

    # https://github.com/triton-lang/triton/issues/1058
    # If B*T*V is too large, program_id * stride will overflow out of int32, so we convert to int64
    program_id = tl.program_id(0).to(tl.int64)

    # 1. Load Y_ptr first because if the target is ignore_index, we can return right away
    Y_ptr += program_id * Y_stride
    y = tl.load(Y_ptr)

    # 2. locate the start index
    X_ptr += program_id * X_stride

    if y == ignore_index:
        # set all X_ptr as 0
        for i in range(0, n_cols, BLOCK_SIZE):
            X_offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(X_ptr + X_offsets, 0.0, mask=X_offsets < n_cols)
        # For ignored tokens, set token accuracy to 0
        if RETURN_TOKEN_ACCURACY:
            token_accuracy_ptr += program_id * token_accuracy_stride
            tl.store(token_accuracy_ptr, 0.0)
        if RETURN_PREDICTED_TOKENS:
            predicted_tokens_ptr += program_id * predicted_tokens_stride
            tl.store(predicted_tokens_ptr, -1)
        return

    loss_ptr += program_id * loss_stride
    if RETURN_Z_LOSS:
        z_loss_ptr += program_id * loss_stride
    if RETURN_TOKEN_ACCURACY:
        token_accuracy_ptr += program_id * token_accuracy_stride
    if RETURN_PREDICTED_TOKENS:
        predicted_tokens_ptr += program_id * predicted_tokens_stride

    if HAS_SOFTCAPPING:
        softcap = softcap.to(tl.float32)
    if HAS_WEIGHT:
        sum_non_ignore_weight = sum_non_ignore_weight.to(tl.float32)
        weight_sum = weight_sum.to(tl.float32)

    if HAS_WEIGHT:
        weight_y = tl.load(weight_ptr + y).cast(tl.float32)

    # Online softmax: 2 loads + 1 store (compared with 3 loads + 1 store for the safe softmax)
    # Refer to Algorithm 3 in the paper: https://arxiv.org/pdf/1805.02867

    # 3. [Online softmax] first pass: find max + sum
    m = float("-inf")  # m is the max value. use the notation from the paper
    d = 0.0  # d is the sum. use the notation from the paper
    argmax_idx = 0  # Track the index of the maximum value for token accuracy / predicted tokens computation
    ori_X_y = tl.load(X_ptr + y).cast(tl.float32)  # we need to store the original value of X_y for the loss calculation
    if HAS_SOFTCAPPING:
        ori_X_y = softcap * tanh(ori_X_y / softcap)

    # Label smoothing is a general case of normal cross entropy
    # See the full derivation at https://github.com/linkedin/Liger-Kernel/pull/198#issue-2503665310
    scaled_x_sum = 0.0
    eps = label_smoothing / n_cols

    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(
            X_ptr + X_offsets,
            mask=X_offsets < n_cols,
            other=float("-inf"),
            # Ensure float32 precision for softmax calculation
        ).cast(tl.float32)
        if HAS_SOFTCAPPING:
            X_block = softcap * tanh(X_block / softcap)
            X_block = tl.where(X_offsets < n_cols, X_block, float("-inf"))
        block_max = tl.max(X_block)

        # Track argmax for accuracy / predicted tokens computation
        if RETURN_TOKEN_ACCURACY or RETURN_PREDICTED_TOKENS:
            # Find the index of the maximum value in this block
            is_max_mask = X_block == block_max
            # Mask out invalid indices with a value larger than n_cols
            masked_offsets = tl.where(is_max_mask, X_offsets, n_cols)
            # Get the first (smallest) index where max occurs
            current_block_argmax_idx = tl.min(masked_offsets)

            is_new_max = block_max > m
            argmax_idx = tl.where(is_new_max, current_block_argmax_idx, argmax_idx)

        if label_smoothing > 0:
            # scale X beforehand to avoid overflow
            if HAS_WEIGHT:
                weight_block = tl.load(weight_ptr + X_offsets, mask=X_offsets < n_cols)
                scaled_x_sum += tl.sum(tl.where(X_offsets < n_cols, -eps * X_block * weight_block, 0.0))
            else:
                scaled_x_sum += tl.sum(tl.where(X_offsets < n_cols, -eps * X_block, 0.0))
        m_new = tl.maximum(m, block_max)
        d = d * tl.exp2((m - m_new) * LOG2_E) + tl.sum(tl.exp2((X_block - m_new) * LOG2_E))
        m = m_new

    # log (sum(e^(X_i))) = log (sum(e ^ (max(X) * e ^ (X_i - max(X)))))
    #                    = log (e^(max(X)) * sum(e ^ (X_i - max(X))))
    #                    = max(X) + log (sum(e ^ (X_i - max(X)))) = m + log d
    lse = m + tl.log(d)

    # 4. [Online Softmax] Second pass: compute gradients
    # For 'mean' reduction, gradients are normalized by number of non-ignored elements (N)
    # dx_y = (softmax(x_y) - 1) / N
    # dx_i = softmax(x_i) / N, i != y
    # For label smoothing:
    # dx_i = (softmax(x_i) - label_smoothing / V) / N, V = n_cols, i != y
    # dx_y = (softmax(x_y) - label_smoothing / V - (1 - label_smoothing)) / N
    #      = dx_i - (1 - label_smoothing) / N
    # With Z loss:
    # dx_i = ((1 + 2 * lse_square_scale * lse) * softmax(x_i) - label_smoothing / V) / N, i != y
    # dx_y = dx_i - (1 - label_smoothing) / N
    # For 'sum' reduction, no normalization is applied:
    # dx_y = softmax(x_y) - 1
    # dx_i = softmax(x_i), for i ≠ y
    if HAS_GRADIENTS:
        for i in range(0, n_cols, BLOCK_SIZE):
            X_offsets = i + tl.arange(0, BLOCK_SIZE)
            X_block = tl.load(
                X_ptr + X_offsets,
                mask=X_offsets < n_cols,
                other=float("-inf"),
                # Ensure float32 precision for softmax calculation
            ).cast(tl.float32)
            if HAS_SOFTCAPPING:
                intermediate = tanh(X_block / softcap)
                X_block = softcap * intermediate

            if not HAS_WEIGHT:
                # softmax(x_i)
                X_block = tl.exp2((X_block - m) * LOG2_E) / d
                # derivative of z-loss: 2 * lse_square_scale * lse * softmax(x_i)
                X_block += 2 * lse_square_scale * lse * X_block
                # smoothing term
                X_block += -eps
                # reduction scale
                if reduction == "mean":
                    X_block = X_block / n_non_ignore
            else:
                weight_block = tl.load(weight_ptr + X_offsets, mask=X_offsets < n_cols)
                softmax_X = tl.exp2((X_block - m) * LOG2_E) / d
                # derivative of original_loss
                dloss_ori = (1 - label_smoothing) * softmax_X
                dloss_ori = dloss_ori * weight_y
                # derivative of smooth_loss
                dloss_smooth = eps * (-weight_block + softmax_X * weight_sum)
                # derivative of z-loss
                dz_loss = 2 * lse_square_scale * lse * softmax_X
                # reduction scale
                if reduction == "mean":
                    dloss_ori = dloss_ori / sum_non_ignore_weight
                    dloss_smooth = dloss_smooth / sum_non_ignore_weight
                    # TODO: Implement weighted z_loss. Currently, z_loss is not scaled by weight.
                    dz_loss = dz_loss / n_non_ignore
                # derivative of total_loss
                X_block = dloss_ori + dloss_smooth + dz_loss

            # chain rule softcapping
            # d(softcap * tanh(x / softcap)) = (1 - tanh^2(x / softcap))
            if HAS_SOFTCAPPING:
                X_block = X_block * (1 - intermediate * intermediate)

            tl.store(X_ptr + X_offsets, X_block, mask=X_offsets < n_cols)

        # dx_y correction: recompute the true-class gradient once, in fp32, and overwrite X[y]
        # (replaces the per-element tl.where removed above). The -(1 - label_smoothing) term must
        # be folded in *before* the result is rounded to X's dtype: dx_y = (softmax(x_y) - 1) / N
        # cancels catastrophically as softmax(x_y) -> 1, so a read-modify-write of the value the
        # loop already stored would lose most of the significant bits in bf16/fp16.
        # ori_X_y is the (softcapped) fp32 logit at y, so softmax_X_y matches the loop exactly.
        # Barrier first so the loop's in-place store to X[y] cannot land after this store.
        tl.debug_barrier()
        softmax_X_y = tl.exp2((ori_X_y - m) * LOG2_E) / d
        if not HAS_WEIGHT:
            dx_y = softmax_X_y
            dx_y += 2 * lse_square_scale * lse * dx_y
            dx_y += -eps
            dx_y += -(1 - label_smoothing)
            if reduction == "mean":
                dx_y = dx_y / n_non_ignore
        else:
            dloss_ori_y = (1 - label_smoothing) * softmax_X_y - (1 - label_smoothing)
            dloss_ori_y = dloss_ori_y * weight_y
            dloss_smooth_y = eps * (-weight_y + softmax_X_y * weight_sum)
            dz_loss_y = 2 * lse_square_scale * lse * softmax_X_y
            if reduction == "mean":
                dloss_ori_y = dloss_ori_y / sum_non_ignore_weight
                dloss_smooth_y = dloss_smooth_y / sum_non_ignore_weight
                dz_loss_y = dz_loss_y / n_non_ignore
            dx_y = dloss_ori_y + dloss_smooth_y + dz_loss_y
        if HAS_SOFTCAPPING:
            t_y = ori_X_y / softcap
            dx_y = dx_y * (1 - t_y * t_y)
        tl.store(X_ptr + y, dx_y)

    # We need tl.debug_barrier() to ensure the new result of X_ptr is written as mentioned in
    # https://github.com/triton-lang/triton/blob/ba42a5c68fd0505f8c42f4202d53be0f8d9a5fe0/python/triton/ops/cross_entropy.py#L34
    tl.debug_barrier()

    # 5. Calculate the loss

    # loss = log (softmax(X_y)) = log ((e ^ (X_y - max(X)) / sum(e ^ (X - max(X))))
    #      = (X_y - max(X)) - log(sum(e ^ (X - max(X))))
    #      = X_y - m - log d = X_y - lse
    # sum(e ^ (X - max(X))) must >= 1 because the max term is e ^ 0 = 1
    # So we can safely calculate log (softmax(X_y)) without overflow
    loss = lse - ori_X_y
    if HAS_WEIGHT:
        loss = weight_y * loss

    # Original loss = H(q, p),  with label smoothing regularization = H(q', p) and (label_smoothing / V) = eps
    # H(q', p) = (1 - label_smoothing) * H(q, p) + label_smoothing * H(u, p)
    #          = (1 - label_smoothing) * H(q, p) + eps * sum(logsoftmax(x_i))
    # By using m (global max of xi) and d (sum of e^(xi-m)), we can simplify as:
    #          = (1 - label_smoothing) * H(q, p) + (sum(-eps * x_i) + label_smoothing * (m + logd))
    # Refer to H(q', p) in section 7 of the paper: https://arxiv.org/pdf/1512.00567
    # pytorch: https://github.com/pytorch/pytorch/blob/2981534f54d49fa3a9755c9b0855e7929c2527f0/aten/src/ATen/native/LossNLL.cpp#L516
    # See full derivation at https://github.com/linkedin/Liger-Kernel/pull/198#issuecomment-2333753087
    if label_smoothing > 0:
        if HAS_WEIGHT:
            smooth_loss = scaled_x_sum + eps * lse * weight_sum
        else:
            smooth_loss = scaled_x_sum + label_smoothing * lse
        loss = loss * (1 - label_smoothing) + smooth_loss

    # An auxiliary loss, z_loss
    # Refer to Page14 Loss function section in the paper PaLM: https://www.jmlr.org/papers/v24/22-1144.html
    z_loss = lse_square_scale * lse * lse
    # Normalize the loss by the number of non-ignored elements if reduction is "mean"
    if reduction == "mean":
        if HAS_WEIGHT:
            loss = loss / sum_non_ignore_weight
        else:
            loss = loss / n_non_ignore
        # TODO: Implement weighted z_loss. Currently, z_loss is not scaled by weight.
        z_loss = z_loss / n_non_ignore
    loss += z_loss

    tl.store(loss_ptr, loss)
    if RETURN_Z_LOSS:
        tl.store(z_loss_ptr, z_loss)
    if RETURN_TOKEN_ACCURACY:
        # Store 1.0 if prediction is correct, 0.0 otherwise
        is_correct = 1.0 if argmax_idx == y else 0.0
        tl.store(token_accuracy_ptr, is_correct)
    if RETURN_PREDICTED_TOKENS:
        tl.store(predicted_tokens_ptr, argmax_idx)
