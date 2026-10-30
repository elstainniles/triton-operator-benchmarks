# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: fla/modules/fused_linear_cross_entropy.py:95 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
# Source archive: dependencies and launch adaptation are not executable here.
# The complete original file and dependency snapshots are under upstream/.
import triton
import triton.language as tl


@triton.jit
def cross_entropy_kernel(
    logits,
    lse,
    target,
    loss,
    total,
    ignore_index,
    label_smoothing: tl.constexpr,
    logit_scale: tl.constexpr,
    logit_softcapping: tl.constexpr,
    reduction: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
):
    """
    This kernel computes both cross entropy loss and the gradient of the input.
    We only consider hard label + mean reduction for now.
    Please refer to https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html for the math.

    Args:
        logits:
            Pointer to logits tensor.
        lse:
            Pointer to logsumexp tensor.
        target: Pointer to target tensor.
        loss:
            Pointer to tensor to store the loss.
        V (int):
            The number of columns in the input tensor.
        total (int):
            The number of non-ignored classes.
        ignore_index (int):
            The index to ignore in the target.
        label_smoothing (float):
            The amount of smoothing when computing the loss, where 0.0 means no smoothing.
        reduction (str):
            The string for the reduction to apply
        BV (int):
            The block size for vocab.
    """

    # https://github.com/triton-lang/triton/issues/1058
    # If B*T*V is too large, i_n * stride will overflow out of int32, so we convert to int64
    i_n = tl.program_id(0).to(tl.int64)
    NV = tl.cdiv(V, BV)

    # 1. Load target first because if the target is ignore_index, we can return right away
    b_y = tl.load(target + i_n)

    # 2. locate the start index
    logits += i_n * V

    if b_y == ignore_index:
        # set all x as 0
        for i in range(0, V, BV):
            o_v = i + tl.arange(0, BV)
            tl.store(logits + o_v, 0.0, mask=o_v < V)
        return

    # Online softmax: 2 loads + 1 store (compared with 3 loads + 1 store for the safe softmax)
    # Refer to Algorithm 3 in the paper: https://arxiv.org/pdf/1805.02867

    # 3. [Online softmax] first pass: compute logsumexp
    # we did this in another kernel
    b_l = tl.load(logits + b_y).to(tl.float32) * logit_scale
    if logit_softcapping is not None:
        b_t_y = tanh(b_l / logit_softcapping)
        b_l = logit_softcapping * b_t_y
        # Save the softcap derivative for the target position for use in step 6
        b_softcap_deriv_y = 1.0 - b_t_y * b_t_y
    b_lse = tl.load(lse + i_n)

    # 4. Calculate the loss
    # loss = lse - logits_l
    b_loss = b_lse - b_l

    # Label smoothing is a general case of normal cross entropy
    # See the full derivation at https://github.com/linkedin/Liger-Kernel/pull/198#issue-2503665310
    b_z = 0.0
    eps = label_smoothing / V

    # We need tl.debug_barrier() as mentioned in
    # https://github.com/triton-lang/triton/blob/ba42a5c68fd0505f8c42f4202d53be0f8d9a5fe0/python/triton/ops/cross_entropy.py#L34
    tl.debug_barrier()

    # 5. [Online Softmax] Second pass: compute gradients
    # For 'mean' reduction, gradients are normalized by number of non-ignored elements
    # dx_y = (softmax(x_y) - 1) / N
    # dx_i = softmax(x_i) / N, i != y
    # For label smoothing:
    # dx_i = (softmax(x_y) - label_smoothing / V) / N, i != y
    # dx_y = (softmax(x_y) - label_smoothing / V - (1 - label_smoothing)) / N
    #      = dx_i - (1 - label_smoothing) / N
    for iv in range(0, NV):
        o_v = iv * BV + tl.arange(0, BV)
        b_logits = tl.load(logits + o_v, mask=o_v < V, other=float('-inf')).to(tl.float32) * logit_scale
        if logit_softcapping is not None:
            b_t = tanh(b_logits / logit_softcapping)
            b_capped = logit_softcapping * b_t
        else:
            b_capped = b_logits
        if label_smoothing > 0:
            # scale X beforehand to avoid overflow
            b_z += tl.sum(tl.where(o_v < V, -eps * b_capped, 0.0))
        b_p = (exp(b_capped - b_lse) - eps) * logit_scale
        # d(softcap * tanh(x/softcap))/dx = 1 - tanh(x/softcap)^2
        if logit_softcapping is not None:
            b_p = b_p * (1.0 - b_t * b_t)
        if reduction == "mean":
            b_p = b_p / total
        tl.store(logits + o_v, b_p, mask=o_v < V)

        tl.debug_barrier()

    # Original loss = H(q, p),  with label smoothing regularization = H(q', p) and (label_smoothing / V) = eps
    # H(q', p) = (1 - label_smoothing) * H(q, p) + label_smoothing * H(u, p)
    #          = (1 - label_smoothing) * H(q, p) + eps * sum(logsoftmax(x_i))
    # By using m (global max of xi) and d (sum of e^(xi-m)), we can simplify as:
    #          = (1 - label_smoothing) * H(q, p) + (-sum(x_i * eps) + label_smoothing * (m + logd))
    # Refer to H(q', p) in section 7 of the paper:
    # https://arxiv.org/pdf/1512.00567
    # pytorch:
    # https://github.com/pytorch/pytorch/blob/2981534f54d49fa3a9755c9b0855e7929c2527f0/aten/src/ATen/native/LossNLL.cpp#L516
    # See full derivation at https://github.com/linkedin/Liger-Kernel/pull/198#issuecomment-2333753087
    if label_smoothing > 0:
        b_loss = b_loss * (1 - label_smoothing) + (b_z + label_smoothing * b_lse)

    # 6. Specially handle the i==y case where `dx_y = (softmax(x_y) - (1 - label_smoothing) / N`
    b_l = tl.load(logits + b_y)

    # The correction term also needs the softcap chain rule factor
    if logit_softcapping is not None:
        b_sc_factor = b_softcap_deriv_y
    else:
        b_sc_factor = 1.0

    # Normalize the loss by the number of non-ignored elements if reduction is "mean"
    if reduction == 'mean':
        b_loss = b_loss / total
        b_l += (label_smoothing - 1) / total * logit_scale * b_sc_factor
    else:
        b_l += (label_smoothing - 1) * logit_scale * b_sc_factor

    tl.store(loss + i_n, b_loss)
    tl.store(logits + b_y, b_l)
