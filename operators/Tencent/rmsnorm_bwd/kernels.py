# User-provided original kernel; collection name requested by the contributor.
# Function bodies/signatures/decorators are unchanged. See ../SOURCE_NOTICE.md.
import triton
import triton.language as tl

@triton.jit
def _rmsnorm_bwd_dxdw_fused(
    dy_ptr,  # pointer to the output grads
    dx_ptr,  # pointer to the input grads
    dw_ptr,  # pointer to the partial weights grads
    input_ptr,  # pointer to the input
    weight_ptr,  # pointer to the weights
    rrms_ptr,  # pointer to the rrms
    stride,  # how much to increase the pointer when moving by 1 row
    s,  # number of rows in input_ptr
    d,  # number of columns in input_ptr, hidden dimension
    BLOCK_S: tl.constexpr, D_ROUNDED: tl.constexpr
):
    # Map the program id to the row of all the data should compute
    row_block_idx = tl.program_id(0).to(tl.int64)  # group_size_s * d, group_id
    col_offsets = tl.arange(0, D_ROUNDED)  # full d for each row
    col_mask = col_offsets < d
    w = tl.load(weight_ptr + col_offsets, mask=col_mask, other=0.).to(tl.float32)

    partial_dw = tl.zeros((D_ROUNDED,), dtype=tl.float32)
    start = row_block_idx * BLOCK_S
    end = min(start + BLOCK_S, s)
    for row in range(start, end):
        # Calc the dx and store
        rrms = tl.load(rrms_ptr + row)
        dy = tl.load(dy_ptr + row * stride + col_offsets, mask=col_mask, other=0.).to(tl.float32)
        x = tl.load(input_ptr + row * stride + col_offsets, mask=col_mask, other=0.).to(tl.float32)

        # impl following torch titan
        x_norm = x * rrms
        partial_dw += dy * x_norm

        dxnorm = w * dy
        c1 = tl.sum(x_norm * dxnorm, axis=0) / d
        dx = (dxnorm - x_norm * c1) * rrms
        tl.store(dx_ptr + row * stride + col_offsets, dx, mask=col_mask)

    tl.store(dw_ptr + row_block_idx * d + col_offsets, partial_dw, mask=col_mask)
