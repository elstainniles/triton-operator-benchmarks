# Copyright 2024 LinkedIn Corporation. BSD-2-Clause; see ../LICENSE.
# Exact function/decorators extracted from the pinned upstream module.
# Loaded in that audited module namespace by opbench.liger; not a standalone import.
@triton.jit
def _swiglu_forward_kernel_flat(a_ptr, b_ptr, c_ptr, total_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    # Grid-Stride Loop
    start_idx = pid * BLOCK_SIZE
    stride = num_progs * BLOCK_SIZE

    for idx in tl.range(start_idx, total_elements, stride):
        offsets = idx + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_elements

        a_val = tl.load(a_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        b_val = tl.load(b_ptr + offsets, mask=mask, other=0.0)

        sig_a = tl.sigmoid(a_val)
        silu_a = a_val * sig_a
        res = silu_a.cast(b_val.dtype) * b_val
        tl.store(c_ptr + offsets, res, mask=mask)
