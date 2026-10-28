# Copyright 2024 LinkedIn Corporation. BSD-2-Clause; see ../LICENSE.
# Exact function/decorators extracted from the pinned upstream module.
# Loaded in that audited module namespace by opbench.liger; not a standalone import.
@triton.jit
def _sparsemax_backward_kernel(
    o_ptr, go_ptr, gi_ptr, stride, n_cols, BLOCK_SIZE: tl.constexpr, num_warps: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    o_row = o_ptr + row * stride
    go_row = go_ptr + row * stride
    gi_row = gi_ptr + row * stride

    offs = tl.arange(0, BLOCK_SIZE)

    supp_cnt = tl.zeros((), tl.float32)
    go_sum = tl.zeros((), tl.float32)

    for i in tl.range(0, tl.cdiv(n_cols, BLOCK_SIZE)):
        offs_iter = i * BLOCK_SIZE + offs
        mask_iter = offs_iter < n_cols
        o_val = tl.load(o_row + offs_iter, mask=mask_iter, other=0.0, cache_modifier=".ca").to(tl.float32)
        go_val = tl.load(go_row + offs_iter, mask=mask_iter, other=0.0).to(tl.float32)
        supp = o_val > 0.0
        go_sum += tl.sum(tl.where(supp, go_val, 0.0))
        supp_cnt += tl.sum(supp.to(tl.float32))

    for i in tl.range(0, tl.cdiv(n_cols, BLOCK_SIZE)):
        offs_iter = i * BLOCK_SIZE + offs
        mask_iter = offs_iter < n_cols
        o_val = tl.load(o_row + offs_iter, mask=mask_iter, other=0.0, cache_modifier=".ca").to(tl.float32)
        go_val = tl.load(go_row + offs_iter, mask=mask_iter, other=0.0).to(tl.float32)
        supp = o_val > 0.0
        gi_val = tl.where(
            supp,
            go_val - tl.cast(go_sum / tl.maximum(supp_cnt, 1e-6), gi_row.dtype.element_ty).to(tl.float32),
            0.0,
        )
        tl.store(gi_row + offs_iter, gi_val.to(gi_row.dtype.element_ty), mask=mask_iter, cache_modifier=".wb")
