# Copyright 2024 LinkedIn Corporation. BSD-2-Clause; see ../LICENSE.
# Exact function/decorators extracted from the pinned upstream module.
# Loaded in that audited module namespace by opbench.liger; not a standalone import.
@triton.jit
def _kldiv_kernel_forward(
    y_ptr,  # [B, S], prediction ptr, the kernel expects the prediction in log-space
    y_stride,  # int, prediction stride
    gt_ptr,  # [B, S], ground truth ptr
    gt_stride,  # int, ground truth stride
    loss_ptr,  # [B] or [B, S] if reduction == _REDUCTION_MODE_NONE, output ptr
    loss_stride,  # int, output stride
    n_cols,  # int, number of columns in the input tensor
    eps,
    BLOCK_SIZE: tl.constexpr,
    # no default: torch.compile re-emits this kernel's source in a module without the
    # _REDUCTION_MODE_* globals, so a default naming one raises NameError. Always pass it.
    reduction: tl.constexpr,
    log_target: tl.constexpr = False,
):
    pid = tl.program_id(0).to(tl.int64)
    y_ptr += pid * y_stride
    gt_ptr += pid * gt_stride
    loss_ptr += pid * loss_stride

    base_offsets = tl.arange(0, BLOCK_SIZE)

    # Pin the scalar to fp32. Eager Triton specializes a non-constexpr Python float to
    # fp32, but Inductor specializes it to fp64; mixing an fp64 scalar into the loop body
    # promotes the whole expression -- including tl.log -- to float64, and re-assigns the
    # fp32 loop-carried `loss_sum` to fp64. This cast is a no-op on the eager path.
    eps = eps.to(tl.float32)

    loss_sum = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + base_offsets
        mask = offsets < n_cols
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        y_true = tl.load(gt_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        # KL(y_true || y) = y_true * (log(y_true) - log(y))
        # We compute KL(y_true || y) with y in the log-space
        if not log_target:
            loss = y_true * (tl.log(tl.maximum(y_true, eps)) - y)
        else:
            loss = tl.exp(y_true) * (y_true - y)

        if reduction == _REDUCTION_MODE_NONE:
            tl.store(loss_ptr + offsets, loss, mask=mask)
        else:
            loss_sum += tl.sum(loss, axis=0)

    if reduction != _REDUCTION_MODE_NONE:
        tl.store(loss_ptr, loss_sum)
