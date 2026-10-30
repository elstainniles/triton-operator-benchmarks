# coding=utf-8
import sys
import os
import pytest
import triton
import torch
import torch.nn.functional as F
import triton.language as tl
import torch_npu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@triton.jit
def exp(x):
    return tl.exp(x.to(tl.float32))

@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def chunk_mesa_net_h_kv_bwd_intra_kernel(
    q_star,
    k,
    v,
    beta,
    h_kv,
    g,
    do,
    dh_kv,
    dq,
    dk_beta,
    dg,
    dv,
    cu_seqlens,
    chunk_indices,
    B: tl.constexpr,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T

    v += (bos * H + i_h) * V
    do += (bos * H + i_h) * V
    h_kv += (i_tg * H + i_h).to(tl.int64) * K*V
    dh_kv += (i_tg * H + i_h).to(tl.int64) * K*V
    q_star += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    beta += (bos * H + i_h)
    g += bos * H + i_h
    dg += bos * H + i_h
    dq += (bos * H + i_h) * K
    dk_beta += (bos * H + i_h) * K
    dv += (bos * H + i_h) * V

    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_ds = tl.zeros([BT, BT], dtype=tl.float32)
    b_dv = tl.zeros([BT, BK], dtype=tl.float32)
    b_dg_last = tl.zeros([1], dtype=tl.float32)
    b_dg = tl.zeros([BT], dtype=tl.float32)

    p_q = tl.make_block_ptr(q_star, (T, K), (H*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_v = tl.make_block_ptr(v, (T, V), (H*V, 1), (i_t * BT, 0), (BT, BV), (1, 0))
    p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_beta = tl.make_block_ptr(beta, (T, ), (H, ), (i_t * BT,), (BT,), (0,))
    p_do = tl.make_block_ptr(do, (T, V), (H*V, 1), (i_t * BT, 0), (BT, BV), (1, 0))
    p_h = tl.make_block_ptr(h_kv, (V, K), (1, V), (0, 0), (BV, BK), (0, 1))
    p_dh = tl.make_block_ptr(dh_kv, (V, K), (1, V), (0, 0), (BV, BK), (0, 1))

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_g = tl.load(p_g, boundary_check=(0,))
    b_beta = tl.load(p_beta, boundary_check=(0, ))
    b_do = tl.load(p_do, boundary_check=(0, 1))
    b_h = tl.load(p_h, boundary_check=(0, 1))
    b_dh = tl.load(p_dh, boundary_check=(0, 1))
    b_g_last = tl.load(g + (min(i_t * BT + BT, T) - 1) * H)

    b_dg_last += tl.sum(b_h * b_dh)
    b_dg_last *= exp(b_g_last)

    b_m = tl.where((o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t[None, :]), exp(b_g[:, None] - b_g[None, :]), 0)
    b_k = (b_k * b_beta[:, None]).to(b_k.dtype)
    b_s = tl.dot(b_q, tl.trans(b_k)) * b_m

    b_ds = tl.dot(b_do, tl.trans(b_v))
    b_dm = b_s * b_ds
    b_dm = tl.where(tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :], b_dm, 0)

    b_dg += tl.sum(b_dm, axis=1)
    b_dg -= tl.sum(b_dm, axis=0)

    b_g_exp_q = exp(b_g)
    b_g_exp_k = tl.where(m_t, exp(-b_g + b_g_last), 0)
    b_ds = b_ds * b_m
    b_dq += tl.dot(b_do, b_h.to(b_do.dtype)) * b_g_exp_q[:, None]
    b_dk += tl.dot(b_v, b_dh.to(b_v.dtype)) * b_g_exp_k[:, None]
    b_dg_last += tl.sum(b_dk * b_k)
    b_dg -= tl.sum(b_dk * b_k, axis=1)
    b_dg += tl.sum(b_dq * b_q, axis=1)
    b_dq += tl.dot(b_ds.to(b_k.dtype), b_k)
    b_dv += tl.dot(b_k, tl.trans(b_dh).to(b_k.dtype)) * b_g_exp_k[:, None] + tl.dot(tl.trans(b_s.to(b_do.dtype)), b_do)
    b_dk += tl.dot(tl.trans(b_ds.to(b_q.dtype)), b_q)

    b_dg = tl.where(o_t < min(i_t * BT + BT, T) - 1, b_dg, b_dg + b_dg_last)
    p_dq = tl.make_block_ptr(dq, (T, K), (H*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk_beta, (T, K), (H*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_dv = tl.make_block_ptr(dv, (T, V), (H*V, 1), (i_t * BT, 0), (BT, BV), (1, 0))
    p_dg = tl.make_block_ptr(dg, (T,), (H,), (i_t * BT,), (BT,), (0,))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))


def generate_test_data(dtype=torch.float32, device='cpu'):
    torch.manual_seed(42)

    B, T, H, K, V = 2, 64, 8, 64, 64
    chunk_size = 16
    IS_VARLEN = False

    q_star = torch.rand(B, T, H, K, dtype=dtype, device=device) / 10
    k = F.normalize(torch.rand(B, T, H, K, dtype=torch.float32, device=device), p=2, dim=-1).to(dtype)
    v = torch.rand(B, T, H, V, dtype=dtype, device=device) / 10
    beta = torch.rand(B, T, H, dtype=dtype, device=device).sigmoid()
    g = torch.rand(B, T, H, dtype=dtype, device=device).float().uniform_(0.8, 0.99).log()
    do = torch.rand(B, T, H, V, dtype=dtype, device=device)

    NT = triton.cdiv(T, chunk_size)
    h_kv = torch.rand(NT * B, H, K, V, dtype=torch.float32, device=device)
    dh_kv = torch.rand(NT * B, H, K, V, dtype=torch.float32, device=device)

    dq = torch.empty(B, T, H, K, dtype=torch.float32, device=device)
    dk = torch.empty(B, T, H, K, dtype=k.dtype, device=device)
    dv = torch.empty(B, T, H, V, dtype=v.dtype, device=device)
    dg = torch.empty(B, T, H, dtype=g.dtype, device=device)

    cu_seqlens = None
    chunk_indices = None

    input_data = {
        'q_star': q_star,
        'k': k,
        'v': v,
        'beta': beta,
        'h_kv': h_kv,
        'dh_kv': dh_kv,
        'g': g,
        'do': do,
        'dq': dq,
        'dk': dk,
        'dv': dv,
        'dg': dg,
        'cu_seqlens': cu_seqlens if cu_seqlens is not None else torch.empty(0, dtype=torch.int32),
        'chunk_indices': chunk_indices if chunk_indices is not None else torch.empty(0, dtype=torch.int32),
        'B': B,
        'T': T,
        'H': H,
        'K': K,
        'V': V,
        'BT': chunk_size,
        'BK': max(triton.next_power_of_2(K), 16),
        'BV': max(triton.next_power_of_2(V), 16),
        'IS_VARLEN': IS_VARLEN,
    }

    grid = (NT, B * H)

    return input_data, grid


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


def get_tolerances(dtype):
    if dtype == torch.float16:
        return 1e-3, 1e-3
    return 1e-5, 1e-5


def compare_output(npu_output: torch.Tensor, gpu_output: torch.Tensor, name: str):
    atol, rtol = get_tolerances(gpu_output.dtype)
    npu_cpu = npu_output.cpu().to(torch.float32)
    gpu_cpu = gpu_output.to(torch.float32)

    max_abs_diff = (npu_cpu - gpu_cpu).abs().max().item()
    max_rel_diff = (max_abs_diff / (gpu_cpu.abs().max().item() + 1e-12))

    passed = torch.allclose(npu_output.cpu(), gpu_output, atol=atol, rtol=rtol)

    print(f"  {name}: dtype={gpu_output.dtype}, atol={atol:.0e}, rtol={rtol:.0e}, "
          f"max_abs_diff={max_abs_diff:.6e}, max_rel_diff={max_rel_diff:.6e}, "
          f"{'PASS' if passed else 'FAIL'}")
    return passed


def run_npu_test_from_pt(ptfile_path):
    try:
        data = torch.load(ptfile_path, map_location=torch.device('cpu'), weights_only=False)
    except Exception as e:
        pytest.fail(f"load file {ptfile_path} failed: {str(e)}")

    input_data = convert_tensor_with_device_type(data["input_data"], device_type='npu')

    chunk_mesa_net_h_kv_bwd_intra_kernel[data["grid"]](
        q_star=input_data['q_star'],
        k=input_data['k'],
        v=input_data['v'],
        beta=input_data['beta'],
        h_kv=input_data['h_kv'],
        g=input_data['g'],
        do=input_data['do'],
        dh_kv=input_data['dh_kv'],
        dq=input_data['dq'],
        dk_beta=input_data['dk'],
        dg=input_data['dg'],
        dv=input_data['dv'],
        cu_seqlens=input_data['cu_seqlens'] if input_data['cu_seqlens'].numel() > 0 else None,
        chunk_indices=input_data['chunk_indices'] if input_data['chunk_indices'].numel() > 0 else None,
        B=input_data['B'],
        T=input_data['T'],
        H=input_data['H'],
        K=input_data['K'],
        V=input_data['V'],
        BT=input_data['BT'],
        BK=input_data['BK'],
        BV=input_data['BV'],
        num_stages=1,
        num_warps=4,
    )

    gpu_output = data["gpu_output"]
    all_passed = True

    for name in ['dq', 'dk', 'dv', 'dg']:
        npu_out = input_data[name].cpu()
        gpu_out = gpu_output[name]
        if not compare_output(npu_out, gpu_out, name):
            all_passed = False

    return all_passed


def run_npu_test_self_consistent(dtype=torch.float32):
    input_data, grid = generate_test_data(dtype=dtype, device='cpu')

    input_data_npu = convert_tensor_with_device_type(input_data, device_type='npu')

    chunk_mesa_net_h_kv_bwd_intra_kernel[grid](
        q_star=input_data_npu['q_star'],
        k=input_data_npu['k'],
        v=input_data_npu['v'],
        beta=input_data_npu['beta'],
        h_kv=input_data_npu['h_kv'],
        g=input_data_npu['g'],
        do=input_data_npu['do'],
        dh_kv=input_data_npu['dh_kv'],
        dq=input_data_npu['dq'],
        dk_beta=input_data_npu['dk'],
        dg=input_data_npu['dg'],
        dv=input_data_npu['dv'],
        cu_seqlens=input_data_npu['cu_seqlens'] if input_data_npu['cu_seqlens'].numel() > 0 else None,
        chunk_indices=input_data_npu['chunk_indices'] if input_data_npu['chunk_indices'].numel() > 0 else None,
        B=input_data_npu['B'],
        T=input_data_npu['T'],
        H=input_data_npu['H'],
        K=input_data_npu['K'],
        V=input_data_npu['V'],
        BT=input_data_npu['BT'],
        BK=input_data_npu['BK'],
        BV=input_data_npu['BV'],
        num_stages=1,
        num_warps=4,
    )

    print(f"\nNPU {dtype} self-consistent test (B={input_data['B']}, T={input_data['T']}, "
          f"H={input_data['H']}, K={input_data['K']}, V={input_data['V']}, BT={input_data['BT']}, "
          f"BK={input_data['BK']}, BV={input_data['BV']}):")
    for name in ['dq', 'dk', 'dv', 'dg']:
        out = input_data_npu[name].cpu()
        has_nan = torch.isnan(out).any().item()
        has_inf = torch.isinf(out).any().item()
        print(f"  {name}: shape={out.shape}, dtype={out.dtype}, "
              f"nan={has_nan}, inf={has_inf}, "
              f"min={out.float().min().item():.6e}, max={out.float().max().item():.6e}")

    return not (has_nan or has_inf)


#def test_chunk_mesa_net_h_kv_bwd_intra_kernel_npu_fp32():
#    print("=" * 60)
#    print("NPU FP32 Test: NPU output vs GPU fp32 output")
#    print("Tolerance: atol=1e-5, rtol=1e-5")
#    print("=" * 60)
#    ptfile = 'test_chunk_mesa_net_h_kv_bwd_intra_kernel_v2.pt'
#    if os.path.exists(ptfile):
#        passed = run_npu_test_from_pt(ptfile)
#        assert passed, "NPU fp32 test FAILED: output mismatch with GPU"
#    else:
#        print(f"  {ptfile} not found, running self-consistent test")
#        passed = run_npu_test_self_consistent(dtype=torch.float32)
#        assert passed, "NPU fp32 self-consistent test FAILED: NaN/Inf detected"


def test_chunk_mesa_net_h_kv_bwd_intra_kernel_npu_fp16():
    print("=" * 60)
    print("NPU FP16 Test: NPU output vs GPU fp16 output")
    print("Tolerance: fp32 outputs atol=1e-5/rtol=1e-5, fp16 outputs atol=1e-3/rtol=1e-3")
    print("  dq/dg: fp32 -> atol=1e-5, rtol=1e-5")
    print("  dk/dv: fp16 -> atol=1e-3, rtol=1e-3")
    print("=" * 60)
    ptfile = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_mesa_net_h_kv_bwd_intra_kernel/test_chunk_mesa_net_h_kv_bwd_intra_kernel_fp16_v2.pt'
    if os.path.exists(ptfile):
        passed = run_npu_test_from_pt(ptfile)
        assert passed, "NPU fp16 test FAILED: output mismatch with GPU"
    else:
        print(f"  {ptfile} not found, running self-consistent test")
        passed = run_npu_test_self_consistent(dtype=torch.float16)
        assert passed, "NPU fp16 self-consistent test FAILED: NaN/Inf detected"


if __name__ == '__main__':
    test_chunk_mesa_net_h_kv_bwd_intra_kernel_npu_fp16()
