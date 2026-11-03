import os

import math

import torch

import torch_npu

import triton

from yb_a87c34e2ef2580 import parallel_deltaformer_bwd_kernel_u
from yb_a87c34e2ef2580 import ParallelDeltaformerFunction

NPU_ID = os.environ.get("BENCH_NPU_ID", "0")

torch.npu.set_device(int(NPU_ID))

def ref_bwd_kernel_u(q, k, grad_v, lse, beta, fa_scale):
    C, H, D = q.shape
    T = k.shape[0]
    q_f = q.float()
    k_f = k.float()
    gv_f = grad_v.float()
    lse_f = lse.float()
    beta_f = beta.float()
    grad_u = torch.zeros(C, H, D, dtype=torch.float32)
    for h in range(H):
        qk = q_f[:, h, :] @ k_f[:, h, :].T * fa_scale
        p = torch.exp2(qk - lse_f[None, :, h]) * beta_f[None, :, h]
        grad_u[:, h, :] = p @ gv_f[:, h, :]
    return grad_u

ATOL, RTOL = 1e-5, 1e-5

def compare_tensors(a, b, name, atol, rtol):
    a_f = a.float()
    b_f = b.float()
    diff = (a_f - b_f).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    denom = b_f.abs().clamp(min=1e-7)
    rel_err = (diff / denom).max().item()
    try:
        torch.testing.assert_close(a_f, b_f, atol=atol, rtol=rtol, equal_nan=True)
        status = "PASS"
    except AssertionError:
        status = "FAIL"
    print(f"    [{name}] {status} | max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}, max_rel_err={rel_err:.2e} | atol={atol:.0e}, rtol={rtol:.0e}")
    return status == "PASS"

def test_bwd_kernel_u():
    print("=" * 80)
    print("Testing parallel_deltaformer_bwd_kernel_u (dtype=float32)")
    print("=" * 80)

    B, T, H, D = 4, 1024, 4, 128
    C = min(512, T)
    dtype = torch.float32
    fa_scale = 1.0 / math.sqrt(D) / math.log(2)

    torch.manual_seed(42)
    qo = torch.randn(1, T, H, D, dtype=dtype, device='npu')
    ko = torch.randn(1, T, H, D, dtype=dtype, device='npu')
    vo = torch.randn(1, T, H, D, dtype=dtype, device='npu')
    betao = torch.randn(1, T, H, dtype=dtype, device='npu').sigmoid()

    u_full, ws, lses = ParallelDeltaformerFunction._forward_impl(qo, ko, vo, betao, C, need_aux=True)
    torch.npu.synchronize()

    lse_full = lses[0].cpu()
    qo_cpu = qo[0].cpu()
    ko_cpu = ko[0].cpu()
    betao_cpu = betao[0].cpu()

    i = 0
    Ci = C
    q_bwd = ko_cpu[i:i + Ci]
    k_bwd = qo_cpu[i + Ci:]
    lse_bwd = lse_full[i + Ci:]
    beta_bwd = betao_cpu[i + Ci:]
    grad_v_bwd = torch.randn(T - Ci, H, D, dtype=dtype)

    T_bwd = T - Ci

    print(f"  Shapes: q={list(q_bwd.shape)}, k={list(k_bwd.shape)}, grad_v={list(grad_v_bwd.shape)}, lse={list(lse_bwd.shape)}, beta={list(beta_bwd.shape)}")
    print(f"  fa_scale={fa_scale:.6f}, lse range=[{lse_bwd.min().item():.4e}, {lse_bwd.max().item():.4e}]")

    cpu_ref = ref_bwd_kernel_u(q_bwd, k_bwd, grad_v_bwd, lse_bwd, beta_bwd, fa_scale)
    print(f"  CPU ref: range=[{cpu_ref.min().item():.4e}, {cpu_ref.max().item():.4e}], has_inf={cpu_ref.isinf().any().item()}")

    q_npu = q_bwd.npu()
    k_npu = k_bwd.npu()
    grad_v_npu = grad_v_bwd.npu()
    lse_npu = lse_bwd.npu()
    beta_npu = beta_bwd.npu()
    grad_u_npu = torch.empty(Ci, H, D, device='npu')

    def grid(META):
        return (triton.cdiv(Ci, META['BLOCK_C']), H)

    parallel_deltaformer_bwd_kernel_u[grid](
        grad_u_npu, q_npu, k_npu, grad_v_npu, lse_npu, beta_npu,
        H, T_bwd, Ci, D, fa_scale,
    )
    torch.npu.synchronize()
    npu_out = grad_u_npu.cpu()
    print(f"  NPU out: range=[{npu_out.min().item():.4e}, {npu_out.max().item():.4e}], has_inf={npu_out.isinf().any().item()}")

    return compare_tensors(npu_out, cpu_ref, "grad_u", ATOL, RTOL)
