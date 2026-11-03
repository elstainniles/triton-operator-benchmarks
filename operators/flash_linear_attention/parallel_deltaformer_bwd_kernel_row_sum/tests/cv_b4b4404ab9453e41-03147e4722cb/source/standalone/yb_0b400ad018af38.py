import os

import math

import torch

import torch_npu

import triton

from yb_a87c34e2ef2580 import parallel_deltaformer_bwd_kernel_row_sum
from yb_a87c34e2ef2580 import ParallelDeltaformerFunction

NPU_ID = os.environ.get("BENCH_NPU_ID", "0")

torch.npu.set_device(int(NPU_ID))

def ref_bwd_kernel_row_sum(q, k, grad_v, u, lse, fa_scale):
    T, H, D = q.shape
    q_f = q.float()
    k_f = k.float()
    gv_f = (-grad_v).float()
    u_f = u.float()
    lse_f = lse.float()
    row_dot_sum = torch.zeros(T, H, dtype=torch.float32)
    for h in range(H):
        qk = q_f[:, h, :] @ k_f[:, h, :].T * fa_scale
        p = torch.exp2(qk - lse_f[:, h][:, None])
        dp = gv_f[:, h, :] @ u_f[:, h, :].T
        causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=0)
        p = p.masked_fill(causal_mask, 0)
        dp = dp.masked_fill(causal_mask, 0)
        row_dot_sum[:, h] = (p * dp).sum(dim=1)
    return row_dot_sum

ATOL, RTOL = 1e-5, 1e-5

def compare_tensors(a, b, name, atol, rtol):
    a_f = a.float()
    b_f = b.float()
    diff = (a_f - b_f).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    denom = b_f.abs().clamp(min=1e-7)
    rel_err = (diff / denom).max().item()
    torch.testing.assert_close(a_f, b_f, atol=atol, rtol=rtol, equal_nan=True)

def test_bwd_kernel_row_sum():
    print("=" * 80)
    print("Testing parallel_deltaformer_bwd_kernel_row_sum (dtype=float32)")
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
    u_full_0 = u_full[0].cpu()
    qo_cpu = qo[0].cpu()
    ko_cpu = ko[0].cpu()

    q_bwd = qo_cpu
    k_bwd = ko_cpu
    u_bwd = u_full_0
    lse_bwd = lse_full
    grad_v_bwd = torch.randn(T, H, D, dtype=dtype)

    print(f"  Shapes: q={list(q_bwd.shape)}, k={list(k_bwd.shape)}, grad_v={list(grad_v_bwd.shape)}, u={list(u_bwd.shape)}, lse={list(lse_bwd.shape)}")
    print(f"  fa_scale={fa_scale:.6f}, lse range=[{lse_bwd.min().item():.4e}, {lse_bwd.max().item():.4e}]")

    cpu_ref = ref_bwd_kernel_row_sum(q_bwd, k_bwd, grad_v_bwd, u_bwd, lse_bwd, fa_scale)
    print(f"  CPU ref: range=[{cpu_ref.min().item():.4e}, {cpu_ref.max().item():.4e}]")

    q_npu = q_bwd.npu()
    k_npu = k_bwd.npu()
    grad_v_npu = grad_v_bwd.npu()
    u_npu = u_bwd.npu()
    lse_npu = lse_bwd.npu()
    row_dot_npu = torch.empty(T, H, dtype=torch.float32, device='npu')

    def grid(META):
        return (triton.cdiv(T, META['BLOCK_C']), H)

    parallel_deltaformer_bwd_kernel_row_sum[grid](
        row_dot_npu, q_npu, k_npu, grad_v_npu, u_npu, lse_npu,
        H, T, D, fa_scale,
    )
    torch.npu.synchronize()
    npu_out = row_dot_npu.cpu()
    print(f"  NPU out: range=[{npu_out.min().item():.4e}, {npu_out.max().item():.4e}]")

    return compare_tensors(npu_out, cpu_ref, "row_dot_sum", ATOL, RTOL)
