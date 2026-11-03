import os, sys, torch, triton
import triton.language as tl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from op_impl import chunk_gsa_bwd_k_kernel_dA
from torch_kernels import chunk_gsa_bwd_k_kernel_dA_torch

if os.environ.get('FLA_USE_FAST_OPS', '0') == '1':
    import triton.language.extra.libdevice as tldevice
    @triton.jit
    def exp(x): return tldevice.fast_expf(x.to(tl.float32))
else:
    @triton.jit
    def exp(x): return tl.exp(x.to(tl.float32))


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({})], key=['BT'], **{})
@triton.jit
def chunk_gsa_bwd_k_kernel_dA_fix1(
    v, g, do, dA,
    chunk_indices, cu_seqlens, scale,
    T, B: tl.constexpr, HQ: tl.constexpr, H: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BC: tl.constexpr, BV: tl.constexpr,
    NC: tl.constexpr, NG: tl.constexpr, IS_VARLEN: tl.constexpr,
):
    i_v, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // NG
    i_t, i_i, i_j = i_c // (NC * NC), (i_c % (NC * NC)) // NC, (i_c % (NC * NC)) % NC
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        all = T
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
        all = B * T

    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V

    if i_t * BT + i_i * BC >= T:
        return

    p_dA = tl.make_block_ptr(dA+((i_v*all+bos)*HQ+i_hq)*BT, (T, BT), (HQ*BT, 1), (i_t*BT+i_i*BC, i_j*BC), (BC, BC), (1, 0))

    b_dA = tl.zeros([BC, BC], dtype=tl.float32)
    if i_i > i_j:
        p_gn = g + (bos + i_t*BT + i_i*BC) * H*V + i_h * V + o_v
        b_gn = tl.load(p_gn, mask=m_v, other=0.)

        p_g = tl.make_block_ptr(g + (bos*H+i_h) * V, (T, V), (H*V, 1), (i_t*BT + i_i*BC, i_v*BV), (BC, BV), (1, 0))
        p_do = tl.make_block_ptr(do + (bos*HQ+i_hq) * V, (T, V), (HQ*V, 1), (i_t*BT + i_i*BC, i_v*BV), (BC, BV), (1, 0))
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_do = (b_do * exp(b_g - b_gn[None, :])).to(tl.float32)

        o_j = tl.arange(0, BC)
        m_j = (i_t * BT + i_j * BC + o_j) < T
        for j in range(0, BC):
            p_v = v + (bos + i_t*BT + i_j*BC + j) * H*V + i_h * V + o_v
            p_gv = g + (bos + i_t*BT + i_j*BC + j) * H*V + i_h * V + o_v
            b_v = tl.load(p_v, mask=m_v, other=0.).to(tl.float32)
            b_gv = tl.load(p_gv, mask=m_v, other=0.).to(tl.float32)
            b_vg_j = b_v * exp(b_gn - b_gv)
            b_dA_j = tl.sum(b_do * b_vg_j[None, :], 1) * scale
            b_dA = tl.where((o_j == j)[None, :], b_dA_j[:, None], b_dA)

        b_dA = tl.where(m_j[None, :], b_dA, 0.)

    elif i_i == i_j:
        p_g = tl.make_block_ptr(g + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t*BT + i_i*BC, i_v*BV), (BC, BV), (1, 0))
        p_do = tl.make_block_ptr(do + (bos*HQ + i_hq) * V, (T, V), (HQ*V, 1), (i_t*BT + i_i*BC, i_v*BV), (BC, BV), (1, 0))
        p_v = v + (bos + i_t*BT + i_j*BC) * H*V + i_h * V + o_v
        p_gv = g + (bos + i_t*BT + i_j*BC) * H*V + i_h * V + o_v
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1)) * scale
        m_v = o_v < V

        o_i = tl.arange(0, BC)
        m_dA = o_i[:, None] >= o_i[None, :]
        for j in range(0, min(BC, T - i_t * BT - i_j * BC)):
            b_v = tl.load(p_v, mask=m_v, other=0).to(tl.float32)
            b_gv = tl.load(p_gv, mask=m_v, other=0).to(tl.float32)
            b_dAj = tl.sum(b_do * b_v[None, :] * exp(b_g - b_gv[None, :]), 1)
            b_dA = tl.where((o_i == j)[None, :], b_dAj[:, None], b_dA)

            p_v += H*V
            p_gv += H*V
        b_dA = tl.where(m_dA, b_dA, 0.)
    tl.store(p_dA, b_dA.to(dA.dtype.element_ty), boundary_check=(0, 1))


def run_kernel(kernel_fn, v, g, do, B, T, HQ, H, V, BT, BC, BV, NG, scale):
    NT = triton.cdiv(T, BT)
    NC = triton.cdiv(BT, BC)
    NV = triton.cdiv(V, BV)
    dA_raw = do.new_empty(NV, B, T, HQ, BT)
    grid = (NV, NT * NC * NC, B * HQ)
    kernel_fn[grid](
        v, g, do, dA_raw,
        cu_seqlens=None, chunk_indices=None,
        scale=scale, T=T, B=B, HQ=HQ, H=H, V=V, BT=BT, BC=BC, BV=BV, NC=NC, NG=NG,
    )
    return dA_raw.sum(0, dtype=dA_raw.dtype)


def compare(name, dA_test, dA_ref):
    diff = (dA_test - dA_ref).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    rel_diff = diff / (dA_ref.abs() + 1e-7)
    max_rel = rel_diff.max().item()
    allclose = torch.allclose(dA_test, dA_ref, rtol=1e-3, atol=1e-3)

    B_dim, T_dim, HQ_dim, BT_dim = dA_test.shape
    BC_val = 16
    BT_val = 64
    NC_val = BT_val // BC_val
    i_b, i_hq = 0, 0

    diag_max = 0.0
    offdiag_max = 0.0
    offdiag_shift_ok = 0
    offdiag_shift_total = 0
    for i_t in range(T_dim // BT_val):
        t_start = i_t * BT_val
        for i_i in range(NC_val):
            for i_j in range(i_i + 1):
                for ii in range(BC_val):
                    for jj in range(0, BC_val, 4):
                        t_i = t_start + i_i * BC_val + ii
                        d = abs(dA_test[i_b, t_i, i_hq, i_j*BC_val+jj].item() - dA_ref[i_b, t_i, i_hq, i_j*BC_val+jj].item())
                        if i_i == i_j:
                            diag_max = max(diag_max, d)
                        else:
                            offdiag_max = max(offdiag_max, d)

    if offdiag_max > 1.0:
        for i_t in range(T_dim // BT_val):
            t_start = i_t * BT_val
            for i_i in range(1, NC_val):
                for i_j in range(i_i):
                    for ii in range(8, BC_val):
                        ii_gpu = ii - 6
                        if ii_gpu < 0:
                            continue
                        match = True
                        for jj in range(0, BC_val, 4):
                            t_i = t_start + i_i * BC_val + ii
                            t_i_gpu = t_start + i_i * BC_val + ii_gpu
                            if abs(dA_test[i_b, t_i, i_hq, i_j*BC_val+jj].item() - dA_ref[i_b, t_i_gpu, i_hq, i_j*BC_val+jj].item()) > 0.1:
                                match = False
                                break
                        offdiag_shift_total += 1
                        if match:
                            offdiag_shift_ok += 1

    status = "PASS" if allclose else "FAIL"
    print(f"\n{'='*60}")
    print(f"  {name}: {status}")
    print(f"{'='*60}")
    print(f"  Max diff:     {max_diff:.6e}")
    print(f"  Mean diff:    {mean_diff:.6e}")
    print(f"  Max rel diff: {max_rel:.6e}")
    print(f"  allclose(1e-3): {allclose}")
    print(f"  Diagonal blocks max diff:    {diag_max:.6e}")
    print(f"  Off-diagonal blocks max diff: {offdiag_max:.6e}")
    if offdiag_shift_total > 0:
        print(f"  Off-diag shift=-6 matches:   {offdiag_shift_ok}/{offdiag_shift_total}")
    return allclose


def test_chunk_gsa_bwd_k_kernel_dA():
    # os.environ['ASCEND_RT_VISIBLE_DEVICES'] = '0'
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'

    data = torch.load('/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_gsa_bwd_k/test_chunk_gsa_bwd_k.pt', map_location='cpu', weights_only=False)
    input_data = data['input_data']

    device = 'npu'
    v = input_data['v'].to(device)
    g = input_data['g'].to(device)
    do_t = input_data['do'].to(device)

    B = input_data['B']
    T = input_data['T']
    HQ = input_data['HQ']
    H = input_data['H']
    V = input_data['V']
    BT = input_data['BT']
    BC = input_data['BC']
    BV = input_data['BV']
    NG = input_data['NG']
    scale = input_data['scale']

    print("Computing torch native reference...")
    dA_torch = chunk_gsa_bwd_k_kernel_dA_torch(v, g, do_t, BT=BT, BC=BC, BV=BV, scale=scale)

    print("\nRunning original Triton kernel...")
    dA_orig = run_kernel(chunk_gsa_bwd_k_kernel_dA, v, g, do_t, B, T, HQ, H, V, BT, BC, BV, NG, scale)
    assert compare("Original Triton kernel", dA_orig, dA_torch)

    print("\nRunning fix1: replace tl.dot with per-column tl.sum loop (i_i > i_j)...")
    try:
        dA_fix1 = run_kernel(chunk_gsa_bwd_k_kernel_dA_fix1, v, g, do_t, B, T, HQ, H, V, BT, BC, BV, NG, scale)
        assert compare("Fix1: tl.sum loop (no tl.dot)", dA_fix1, dA_torch)
    except Exception as e:
        print(f"  Fix1 FAILED: {e}")

    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print("Original:  tl.dot(b_do, b_vg) * scale  -> rows 8-15 shifted by -6 (NPU-IR bug)")
    print("Fix1:      per-column tl.sum loop       -> avoids tl.dot entirely")
    print("Fix2:      tl.dot with forced float32   -> tests if dtype cast is the trigger")

# 单独测试：chunk_gsa_bwd_k_kernel_dA kernel, 0527包测试ok
# https://codehub-y.huawei.com/c00564736/AscendNPU_IR/issues/23
if __name__ == '__main__':
    test_chunk_gsa_bwd_k_kernel_dA()
