# User-provided original kernel; collection name requested by the contributor.
# Function bodies/signatures/decorators are unchanged. See ../SOURCE_NOTICE.md.
import triton
import triton.language as tl

@triton.jit
def _tanh(x):
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _fused_situgluv2_fwd_triton_kernel(
    a_ptr, b_ptr, out_ptr,
    N,
    stride_am, stride_an,
    stride_bm, stride_bn,
    stride_om, stride_on,
    beta,
    inv_beta,
    linear_beta,
    inv_linear_beta,
    BLOCK: tl.constexpr,
):
    m = tl.program_id(0).to(tl.int64)
    nb = tl.program_id(1).to(tl.int64)

    offs_n = nb * BLOCK + tl.arange(0, BLOCK)
    mask = offs_n < N

    a_row = a_ptr + m * stride_am + offs_n * stride_an
    b_row = b_ptr + m * stride_bm + offs_n * stride_bn
    o_row = out_ptr + m * stride_om + offs_n * stride_on

    a = tl.load(a_row, mask=mask, other=0).to(tl.float32)
    b = tl.load(b_row, mask=mask, other=0).to(tl.float32)

    sig_a = tl.sigmoid(a)
    z_a = a * inv_beta
    tanh_a = _tanh(z_a)
    f = beta * tanh_a * sig_a

    z_b = b * inv_linear_beta
    tanh_b = _tanh(z_b)
    r = linear_beta * tanh_b

    out = f * r
    tl.store(o_row, out, mask=mask)
