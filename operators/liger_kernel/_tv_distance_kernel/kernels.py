# Copyright 2024 LinkedIn Corporation. BSD-2-Clause; see ../LICENSE.
# Exact function/decorators extracted from the pinned upstream module.
# Loaded in that audited module namespace by opbench.liger; not a standalone import.
@triton.jit
def _tv_distance_kernel(
    p_ptr,
    p_stride,
    q_ptr,
    q_stride,
    loss_ptr,
    loss_stride,
    grads_ptr,
    grads_stride,
    label_ptr,
    ignore_index: tl.constexpr,
    n_cols,
    scale,  # pre-computed reduction scale for gradients (fused into kernel)
    BLOCK_SIZE: tl.constexpr,
    HAS_LABEL: tl.constexpr,
    # no default: torch.compile re-emits this kernel's source in a module without the
    # _REDUCTION_MODE_* globals, so a default naming one raises NameError. Always pass it.
    reduction: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    p_ptr += pid * p_stride
    q_ptr += pid * q_stride
    loss_ptr += pid * loss_stride
    grads_ptr += pid * grads_stride
    label_ptr += pid

    base_offsets = tl.arange(0, BLOCK_SIZE)

    # Pin the scalar to fp32. Eager Triton specializes a non-constexpr Python float to
    # fp32, but Inductor specializes it to fp64; mixing an fp64 scalar into the loop body
    # promotes the gradient expression to float64 on every iteration and makes the
    # compiled path disagree with eager. This cast is a no-op on the eager path.
    scale = scale.to(tl.float32)

    if HAS_LABEL:
        label = tl.load(label_ptr)
        if label == ignore_index:
            for i in range(0, n_cols, BLOCK_SIZE):
                offsets = i + base_offsets
                mask = offsets < n_cols
                tl.store(grads_ptr + offsets, 0.0, mask=mask)
                if reduction == _REDUCTION_MODE_NONE:
                    tl.store(loss_ptr + offsets, 0.0, mask=mask)
            return

    loss_sum = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + base_offsets
        mask = offsets < n_cols

        p = tl.load(p_ptr + offsets, mask=mask, other=0.0)
        q = tl.load(q_ptr + offsets, mask=mask, other=0.0)

        # TVD(P || Q) = 0.5 * |P - Q|
        tv_loss = 0.5 * tl.abs(p - q)

        # d/dp 0.5 * |p - q| = 0.5 * sgn(p - q), which is 0 where p == q
        # Fuse reduction scaling into gradient computation (eliminates separate Python division)
        grad_res = tl.where(p == q, 0.0, tl.where(p > q, 0.5 * scale, -0.5 * scale))

        tl.store(grads_ptr + offsets, grad_res, mask=mask)

        if reduction == _REDUCTION_MODE_NONE:
            tl.store(loss_ptr + offsets, tv_loss, mask=mask)
        else:
            loss_sum += tl.sum(tv_loss, axis=0)

    if reduction != _REDUCTION_MODE_NONE:
        # Fuse reduction scaling into loss (same scale as gradients; avoids Python division)
        tl.store(loss_ptr, loss_sum * scale)
