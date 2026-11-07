# User-provided original kernel; collection name requested by the contributor.
# Function bodies/signatures/decorators are unchanged. See ../SOURCE_NOTICE.md.
import triton
import triton.language as tl

@triton.jit
def _rmsnorm_fwd_fused(
    input_ptr,  # pointer to the input
    output_ptr,  # pointer to the output
    weight_ptr,  # pointer to the weights
    rrms_ptr,  # pointer to the rrms
    stride,  # how much to increase the pointer when moving by 1 row
    d,  # number of columns in input
    eps,  # epsilon to avoid division by zero
    D_ROUNDED: tl.constexpr
):
    # Map the program id to the row of input and output it should compute.
    row_idx = tl.program_id(0).to(tl.int64)
    output_ptr += row_idx * stride
    input_ptr += row_idx * stride

    # Compute rms
    col_offsets = tl.arange(0, D_ROUNDED)
    mask = col_offsets < d
    x = tl.load(input_ptr + col_offsets, mask=mask, other=0.).to(tl.float32)
    x_square = x * x
    ms = tl.sum(x_square, axis=0) / d + eps
    rrms = 1 / tl.sqrt(ms)

    # Write rms
    tl.store(rrms_ptr + row_idx, rrms)

    # Calc y
    w = tl.load(weight_ptr + col_offsets, mask=mask, other=0.).to(tl.float32)
    x_norm = x * rrms
    y = x_norm * w

    # store data
    tl.store(output_ptr + col_offsets, y, mask=mask)
