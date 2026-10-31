import torch
import triton
import triton.language as tl
import torch_npu
import torch.nn.functional as F
from typing import Optional

def validate_cmp(dtype, y_cal, y_ref, overflow_mode: Optional[str] = None, device_type: Optional[str] = None):
    if device_type is not None:
        target_device = torch.device(device_type)
        y_cal = y_cal.to(target_device)
        y_ref = y_ref.to(target_device)
    else:
        y_cal = y_cal.npu()
        y_ref = y_ref.npu()
    if overflow_mode == "saturate":
        if dtype in ['float32', 'float16']:
            min_value = -torch.finfo(dtype).min
            max_value = torch.finfo(dtype).max
        elif dtype in ['int32', 'int16', 'int8']:
            min_value = torch.iinfo(dtype).min
            max_value = torch.iinfo(dtype).max
        elif dtype == 'bool':
            min_value = 0
            max_value = 1
        else:
            raise ValueError('Invalid parameter "dtype" is found : {}'.format(dtype))
        y_ref = torch.clamp(y_ref, min=min_value, max=max_value)
    if dtype == 'float16':
        torch.testing.assert_close(y_ref, y_cal, rtol=1e-03, atol=1e-03, equal_nan=True)
    elif dtype == 'bfloat16':
        torch.testing.assert_close(y_ref.to(torch.float32), y_cal.to(torch.float32), rtol=1e-02, atol=1e-03, equal_nan=True)
    elif dtype == 'float32':
        torch.testing.assert_close(y_ref, y_cal, rtol=1e-05, atol=1e-05, equal_nan=True)
    elif dtype == 'int32' or dtype == 'int64' or dtype == 'int16' or dtype == 'int8':
        assert torch.equal(y_cal, y_ref)
    elif dtype == 'bool':
        assert torch.equal(y_cal, y_ref)
    else:
        raise ValueError('Invalid parameter "dtype" is found : {}'.format(dtype))

def compare_data_precision(dict_ref: dict, dict_cal: dict, device_type: str):
    keys_ref, keys_cal = set(dict_ref.keys()), set(dict_cal.keys())
    if not keys_ref.issubset(keys_cal):
        raise ValueError("The keys of dict_ref is not subset of dict_cal")
    for key in dict_ref.keys():
        val_a, val_b = dict_ref[key], dict_cal[key]
        if type(val_a) != type(val_b):
            raise ValueError("The data type of two dicts are different")
        if isinstance(val_a, torch.Tensor):
            validate_cmp(dtype=str(val_a.dtype).split('.')[-1], y_ref=val_a, y_cal=val_b, device_type=device_type)
        else:
            raise ValueError("Non-tensor type is not currently supported")

def convert_tensor_with_device_type(indata: dict, device_type: str):
    target_device = torch.device(device_type)
    outdata = {}
    for key, value in indata.items():
        if isinstance(value, torch.Tensor):
            if value.device.type != target_device.type:
                outdata[key] = value.to(target_device)
            else:
                outdata[key] = value
        else:
            outdata[key] = value
    return outdata

@triton.jit
def exp(x): return tl.exp(x.to(tl.float32))
@triton.jit
def exp2(x): return tl.math.exp2(x.to(tl.float32))


IS_TF32_SUPPORTED = False
if IS_TF32_SUPPORTED:
    SOLVE_TRIL_DOT_PRECISION = tl.constexpr('tf32')
else:
    SOLVE_TRIL_DOT_PRECISION = tl.constexpr('ieee')


def prepare_chunk_indices(cu_seqlens, chunk_size):
    cu_seqlens = cu_seqlens.tolist()
    chunk_indices = []
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i], cu_seqlens[i + 1]
        seq_len = end - start
        num_chunks = triton.cdiv(seq_len, chunk_size)
        for j in range(num_chunks):
            chunk_indices.append([i, j])
    return torch.tensor(chunk_indices, dtype=torch.long, device=cu_seqlens.device)


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK}, num_warps=num_warps)
        for BK in [32, 64]
        for num_warps in [1, 2, 4]
    ],
    key=['H', 'HV', 'K', 'BC'],
)
@triton.jit(do_not_specialize=['T'])
def chunk_gated_delta_rule_fwd_kkt_solve_kernel(
    k,
    g,
    beta,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    USE_G: tl.constexpr,
    USE_EXP2: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Fused kernel: compute beta * K @ K^T (lower triangular) + solve_tril (I+A)^{-1} in one pass.

    This kernel fuses chunk_scaled_dot_kkt_fwd and solve_tril into a single kernel,
    avoiding the HBM round-trip for the intermediate A matrix.

    Steps:
    1. Compute all 10 lower-triangular [BC, BC] blocks of beta * K @ K^T in registers
    2. Apply gate and beta scaling
    3. Forward substitution on diagonal blocks
    4. Block merge to get full (I+A)^{-1}
    5. Write result to A (output)
    """
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // HV, i_bh % HV

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if i_t * BT >= T:
        return

    i_tc0 = i_t * BT
    i_tc1 = i_t * BT + BC
    i_tc2 = i_t * BT + 2 * BC
    i_tc3 = i_t * BT + 3 * BC

    k += (bos * H + i_h // (HV // H)) * K
    A += (bos * HV + i_h) * BT

    o_i = tl.arange(0, BC)
    m_tc0 = (i_tc0 + o_i) < T
    m_tc1 = (i_tc1 + o_i) < T
    m_tc2 = (i_tc2 + o_i) < T
    m_tc3 = (i_tc3 + o_i) < T

    # load beta for each sub-chunk
    p_b0 = tl.make_block_ptr(beta + bos * HV + i_h, (T,), (HV,), (i_tc0,), (BC,), (0,))
    p_b1 = tl.make_block_ptr(beta + bos * HV + i_h, (T,), (HV,), (i_tc1,), (BC,), (0,))
    p_b2 = tl.make_block_ptr(beta + bos * HV + i_h, (T,), (HV,), (i_tc2,), (BC,), (0,))
    p_b3 = tl.make_block_ptr(beta + bos * HV + i_h, (T,), (HV,), (i_tc3,), (BC,), (0,))
    b_b0 = tl.load(p_b0, boundary_check=(0,)).to(tl.float32)
    b_b1 = tl.load(p_b1, boundary_check=(0,)).to(tl.float32)
    b_b2 = tl.load(p_b2, boundary_check=(0,)).to(tl.float32)
    b_b3 = tl.load(p_b3, boundary_check=(0,)).to(tl.float32)

    # load gate if used
    if USE_G:
        p_g0 = tl.make_block_ptr(g + bos * HV + i_h, (T,), (HV,), (i_tc0,), (BC,), (0,))
        p_g1 = tl.make_block_ptr(g + bos * HV + i_h, (T,), (HV,), (i_tc1,), (BC,), (0,))
        p_g2 = tl.make_block_ptr(g + bos * HV + i_h, (T,), (HV,), (i_tc2,), (BC,), (0,))
        p_g3 = tl.make_block_ptr(g + bos * HV + i_h, (T,), (HV,), (i_tc3,), (BC,), (0,))

        b_g0 = tl.load(p_g0, boundary_check=(0,)).to(tl.float32)
        b_g1 = tl.load(p_g1, boundary_check=(0,)).to(tl.float32)
        b_g2 = tl.load(p_g2, boundary_check=(0,)).to(tl.float32)
        b_g3 = tl.load(p_g3, boundary_check=(0,)).to(tl.float32)

    ############################################################################
    # Step 1: compute all 10 lower-triangular [BC, BC] blocks of K @ K^T
    ############################################################################

    # 4 diagonal blocks
    b_A00 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A11 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A22 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A33 = tl.zeros([BC, BC], dtype=tl.float32)

    # 6 off-diagonal blocks
    b_A10 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A20 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A21 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A30 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A31 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A32 = tl.zeros([BC, BC], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        p_k0 = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0))
        b_k0 = tl.load(p_k0, boundary_check=(0, 1))
        # diagonal block 0
        b_A00 += tl.dot(b_k0, tl.trans(b_k0))

        if i_tc1 < T:
            p_k1 = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0))
            b_k1 = tl.load(p_k1, boundary_check=(0, 1))
            # diagonal block 1
            b_A11 += tl.dot(b_k1, tl.trans(b_k1))
            # off-diagonal (1,0)
            b_A10 += tl.dot(b_k1, tl.trans(b_k0))

            if i_tc2 < T:
                p_k2 = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0))
                b_k2 = tl.load(p_k2, boundary_check=(0, 1))
                # diagonal block 2
                b_A22 += tl.dot(b_k2, tl.trans(b_k2))
                # off-diagonal (2,0), (2,1)
                b_A20 += tl.dot(b_k2, tl.trans(b_k0))
                b_A21 += tl.dot(b_k2, tl.trans(b_k1))

                if i_tc3 < T:
                    p_k3 = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0))
                    b_k3 = tl.load(p_k3, boundary_check=(0, 1))
                    # diagonal block 3
                    b_A33 += tl.dot(b_k3, tl.trans(b_k3))
                    # off-diagonal (3,0), (3,1), (3,2)
                    b_A30 += tl.dot(b_k3, tl.trans(b_k0))
                    b_A31 += tl.dot(b_k3, tl.trans(b_k1))
                    b_A32 += tl.dot(b_k3, tl.trans(b_k2))

    ############################################################################
    # Step 2: apply gate and beta scaling
    ############################################################################

    # apply gate, beta scaling, and masking
    # m_d: strictly lower triangular mask for diagonal blocks
    # m_tc: boundary mask to prevent NaN from 0 * inf (IEEE 754) when
    #   out-of-bounds g loads as 0 via boundary_check and exp(0 - g_inbounds) overflows
    m_d = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    if USE_G:
        if USE_EXP2:
            b_A00 *= tl.where(m_d & m_tc0[:, None] & m_tc0[None, :], exp2(b_g0[:, None] - b_g0[None, :]), 0.)
            b_A11 *= tl.where(m_d & m_tc1[:, None] & m_tc1[None, :], exp2(b_g1[:, None] - b_g1[None, :]), 0.)
            b_A22 *= tl.where(m_d & m_tc2[:, None] & m_tc2[None, :], exp2(b_g2[:, None] - b_g2[None, :]), 0.)
            b_A33 *= tl.where(m_d & m_tc3[:, None] & m_tc3[None, :], exp2(b_g3[:, None] - b_g3[None, :]), 0.)

            b_A10 *= tl.where(m_tc1[:, None] & m_tc0[None, :], exp2(b_g1[:, None] - b_g0[None, :]), 0.)
            b_A20 *= tl.where(m_tc2[:, None] & m_tc0[None, :], exp2(b_g2[:, None] - b_g0[None, :]), 0.)
            b_A21 *= tl.where(m_tc2[:, None] & m_tc1[None, :], exp2(b_g2[:, None] - b_g1[None, :]), 0.)
            b_A30 *= tl.where(m_tc3[:, None] & m_tc0[None, :], exp2(b_g3[:, None] - b_g0[None, :]), 0.)
            b_A31 *= tl.where(m_tc3[:, None] & m_tc1[None, :], exp2(b_g3[:, None] - b_g1[None, :]), 0.)
            b_A32 *= tl.where(m_tc3[:, None] & m_tc2[None, :], exp2(b_g3[:, None] - b_g2[None, :]), 0.)
        else:
            b_A00 *= tl.where(m_d & m_tc0[:, None] & m_tc0[None, :], exp(b_g0[:, None] - b_g0[None, :]), 0.)
            b_A11 *= tl.where(m_d & m_tc1[:, None] & m_tc1[None, :], exp(b_g1[:, None] - b_g1[None, :]), 0.)
            b_A22 *= tl.where(m_d & m_tc2[:, None] & m_tc2[None, :], exp(b_g2[:, None] - b_g2[None, :]), 0.)
            b_A33 *= tl.where(m_d & m_tc3[:, None] & m_tc3[None, :], exp(b_g3[:, None] - b_g3[None, :]), 0.)

            b_A10 *= tl.where(m_tc1[:, None] & m_tc0[None, :], exp(b_g1[:, None] - b_g0[None, :]), 0.)
            b_A20 *= tl.where(m_tc2[:, None] & m_tc0[None, :], exp(b_g2[:, None] - b_g0[None, :]), 0.)
            b_A21 *= tl.where(m_tc2[:, None] & m_tc1[None, :], exp(b_g2[:, None] - b_g1[None, :]), 0.)
            b_A30 *= tl.where(m_tc3[:, None] & m_tc0[None, :], exp(b_g3[:, None] - b_g0[None, :]), 0.)
            b_A31 *= tl.where(m_tc3[:, None] & m_tc1[None, :], exp(b_g3[:, None] - b_g1[None, :]), 0.)
            b_A32 *= tl.where(m_tc3[:, None] & m_tc2[None, :], exp(b_g3[:, None] - b_g2[None, :]), 0.)
    else:
        b_A00 = tl.where(m_d, b_A00, 0.)
        b_A11 = tl.where(m_d, b_A11, 0.)
        b_A22 = tl.where(m_d, b_A22, 0.)
        b_A33 = tl.where(m_d, b_A33, 0.)

    # diagonal blocks: scaled by beta
    b_A00 = b_A00 * b_b0[:, None]
    b_A11 = b_A11 * b_b1[:, None]
    b_A22 = b_A22 * b_b2[:, None]
    b_A33 = b_A33 * b_b3[:, None]

    # off-diagonal blocks: full block, scaled by beta
    b_A10 = b_A10 * b_b1[:, None]
    b_A20 = b_A20 * b_b2[:, None]
    b_A21 = b_A21 * b_b2[:, None]
    b_A30 = b_A30 * b_b3[:, None]
    b_A31 = b_A31 * b_b3[:, None]
    b_A32 = b_A32 * b_b3[:, None]

    ############################################################################
    # Step 3: forward substitution on diagonal blocks -> (I + A_diag)^{-1}
    #
    # Same algorithm as solve_tril, but rows are extracted from in-register
    # [BC, BC] tensor via tl.sum(tl.where(mask, tensor, 0), 0) instead of
    # tl.load from HBM.
    ############################################################################

    b_Ai00 = -b_A00
    b_Ai11 = -b_A11
    b_Ai22 = -b_A22
    b_Ai33 = -b_A33

    for i in range(2, min(BC, T - i_tc0)):
        b_a00 = tl.sum(tl.where((o_i == i)[:, None], -b_A00, 0.), 0)
        b_a00 = tl.where(o_i < i, b_a00, 0.)
        b_a00 = b_a00 + tl.sum(b_a00[:, None] * b_Ai00, 0)
        b_Ai00 = tl.where((o_i == i)[:, None], b_a00, b_Ai00)
    for i in range(2, min(BC, T - i_tc1)):
        b_a11 = tl.sum(tl.where((o_i == i)[:, None], -b_A11, 0.), 0)
        b_a11 = tl.where(o_i < i, b_a11, 0.)
        b_a11 = b_a11 + tl.sum(b_a11[:, None] * b_Ai11, 0)
        b_Ai11 = tl.where((o_i == i)[:, None], b_a11, b_Ai11)
    for i in range(2, min(BC, T - i_tc2)):
        b_a22 = tl.sum(tl.where((o_i == i)[:, None], -b_A22, 0.), 0)
        b_a22 = tl.where(o_i < i, b_a22, 0.)
        b_a22 = b_a22 + tl.sum(b_a22[:, None] * b_Ai22, 0)
        b_Ai22 = tl.where((o_i == i)[:, None], b_a22, b_Ai22)
    for i in range(2, min(BC, T - i_tc3)):
        b_a33 = tl.sum(tl.where((o_i == i)[:, None], -b_A33, 0.), 0)
        b_a33 = tl.where(o_i < i, b_a33, 0.)
        b_a33 = b_a33 + tl.sum(b_a33[:, None] * b_Ai33, 0)
        b_Ai33 = tl.where((o_i == i)[:, None], b_a33, b_Ai33)

    b_Ai00 += m_I
    b_Ai11 += m_I
    b_Ai22 += m_I
    b_Ai33 += m_I

    ############################################################################
    # Step 4: block merge -> full (I + A)^{-1}
    ############################################################################

    b_Ai10 = -tl.dot(
        tl.dot(b_Ai11, b_A10, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai00,
        input_precision=SOLVE_TRIL_DOT_PRECISION
    )
    b_Ai21 = -tl.dot(
        tl.dot(b_Ai22, b_A21, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai11,
        input_precision=SOLVE_TRIL_DOT_PRECISION
    )
    b_Ai32 = -tl.dot(
        tl.dot(b_Ai33, b_A32, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai22,
        input_precision=SOLVE_TRIL_DOT_PRECISION
    )

    b_Ai20 = -tl.dot(
        b_Ai22,
        tl.dot(b_A20, b_Ai00, input_precision=SOLVE_TRIL_DOT_PRECISION) +
        tl.dot(b_A21, b_Ai10, input_precision=SOLVE_TRIL_DOT_PRECISION),
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai31 = -tl.dot(
        b_Ai33,
        tl.dot(b_A31, b_Ai11, input_precision=SOLVE_TRIL_DOT_PRECISION) +
        tl.dot(b_A32, b_Ai21, input_precision=SOLVE_TRIL_DOT_PRECISION),
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai30 = -tl.dot(
        b_Ai33,
        tl.dot(b_A30, b_Ai00, input_precision=SOLVE_TRIL_DOT_PRECISION) +
        tl.dot(b_A31, b_Ai10, input_precision=SOLVE_TRIL_DOT_PRECISION) +
        tl.dot(b_A32, b_Ai20, input_precision=SOLVE_TRIL_DOT_PRECISION),
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )

    ############################################################################
    # Step 5: store full (I + A)^{-1} to output A
    ############################################################################

    p_A00 = tl.make_block_ptr(A, (T, BT), (HV*BT, 1), (i_tc0, 0), (BC, BC), (1, 0))
    p_A10 = tl.make_block_ptr(A, (T, BT), (HV*BT, 1), (i_tc1, 0), (BC, BC), (1, 0))
    p_A11 = tl.make_block_ptr(A, (T, BT), (HV*BT, 1), (i_tc1, BC), (BC, BC), (1, 0))
    p_A20 = tl.make_block_ptr(A, (T, BT), (HV*BT, 1), (i_tc2, 0), (BC, BC), (1, 0))
    p_A21 = tl.make_block_ptr(A, (T, BT), (HV*BT, 1), (i_tc2, BC), (BC, BC), (1, 0))
    p_A22 = tl.make_block_ptr(A, (T, BT), (HV*BT, 1), (i_tc2, 2*BC), (BC, BC), (1, 0))
    p_A30 = tl.make_block_ptr(A, (T, BT), (HV*BT, 1), (i_tc3, 0), (BC, BC), (1, 0))
    p_A31 = tl.make_block_ptr(A, (T, BT), (HV*BT, 1), (i_tc3, BC), (BC, BC), (1, 0))
    p_A32 = tl.make_block_ptr(A, (T, BT), (HV*BT, 1), (i_tc3, 2*BC), (BC, BC), (1, 0))
    p_A33 = tl.make_block_ptr(A, (T, BT), (HV*BT, 1), (i_tc3, 3*BC), (BC, BC), (1, 0))

    tl.store(p_A00, b_Ai00.to(A.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_A10, b_Ai10.to(A.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_A11, b_Ai11.to(A.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_A20, b_Ai20.to(A.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_A21, b_Ai21.to(A.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_A22, b_Ai22.to(A.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_A30, b_Ai30.to(A.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_A31, b_Ai31.to(A.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_A32, b_Ai32.to(A.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_A33, b_Ai33.to(A.dtype.element_ty), boundary_check=(0, 1))


def chunk_gated_delta_rule_fwd_intra(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
    use_exp2: bool = False,
) -> torch.Tensor:
    B, T, H, K, HV = *k.shape, beta.shape[2]
    BT = chunk_size
    BC = 16

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    A = torch.zeros(B, T, HV, BT, device=k.device, dtype=k.dtype)
    chunk_gated_delta_rule_fwd_kkt_solve_kernel[(NT, B * HV)](
        k=k,
        g=g,
        beta=beta,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        HV=HV,
        K=K,
        BT=BT,
        BC=BC,
	USE_EXP2=use_exp2,
    )
    
    return A


def test_chunk_gated_delta_rule_fwd_kkt_solve_kernel():
    ptfile_path = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_gated_delta_rule_fwd_kkt_solve_kernel/test_chunk_gated_delta_rule_fwd_kkt_solve_kernel_cuda.pt'
    data = torch.load(ptfile_path, map_location=torch.device('cpu'), weights_only=False)
    
    input_data = convert_tensor_with_device_type(data["input_data"], device_type='npu')
    
    k = input_data['k']
    v = input_data['v']
    g = input_data['g']
    beta = input_data['beta']
    chunk_size = input_data['chunk_size']
    
    A = chunk_gated_delta_rule_fwd_intra(
        k=k, v=v, g=g, beta=beta, chunk_size=chunk_size
    )
    
    npu_out = {
        'A': A.cpu(),
    }
    
    compare_data_precision(data["gpu_output"], npu_out, device_type='cpu')


if __name__ == "__main__":
    test_chunk_gated_delta_rule_fwd_kkt_solve_kernel()
