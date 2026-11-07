# User-provided original kernel; collection name requested by the contributor.
# Function bodies/signatures/decorators are unchanged. See ../SOURCE_NOTICE.md.
import triton
import triton.language as tl

@triton.jit
def _fused_situgluv2_bwd_triton_kernel(
    a_ptr, b_ptr, dout_ptr,
    da_ptr, db_ptr,
    N,
    stride_am, stride_an,
    stride_bm, stride_bn,
    stride_dom, stride_don,
    stride_dam, stride_dan,
    stride_dbm, stride_dbn,
    beta, inv_beta,
    linear_beta, inv_linear_beta,
    BLOCK: tl.constexpr,
):
    m = tl.program_id(0).to(tl.int64)
    nb = tl.program_id(1).to(tl.int64)
    offs_n = nb * BLOCK + tl.arange(0, BLOCK)
    mask = offs_n < N

    a_row = a_ptr + m * stride_am + offs_n * stride_an
    b_row = b_ptr + m * stride_bm + offs_n * stride_bn
    do_row = dout_ptr + m * stride_dom + offs_n * stride_don
    da_row = da_ptr + m * stride_dam + offs_n * stride_dan
    db_row = db_ptr + m * stride_dbm + offs_n * stride_dbn

    a = tl.load(a_row, mask=mask, other=0).to(tl.float32)
    b = tl.load(b_row, mask=mask, other=0).to(tl.float32)
    dout = tl.load(do_row, mask=mask, other=0).to(tl.float32)

    s = tl.sigmoid(a)
    s_neg = tl.sigmoid(-a)
    z_a = a * inv_beta
    p_a = tl.sigmoid(2.0 * z_a)
    t_a = 2.0 * p_a - 1.0
    tanh_prime_a = 4.0 * p_a * (1.0 - p_a)
    f = beta * t_a * s
    dfdx = beta * t_a * s * s_neg + s * tanh_prime_a

    z_b = b * inv_linear_beta
    p_b = tl.sigmoid(2.0 * z_b)
    t_b = 2.0 * p_b - 1.0
    tanh_prime_b = 4.0 * p_b * (1.0 - p_b)
    r = linear_beta * t_b

    da = dout * r * dfdx
    db = dout * f * tanh_prime_b

    tl.store(da_row, da, mask=mask)
    tl.store(db_row, db, mask=mask)
