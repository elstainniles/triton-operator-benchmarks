
from opbench.fla_fixtures import load_fixture
import os
import math
import torch
import torch_npu
import pytest
NPU_ID = os.environ.get('BENCH_NPU_ID', '0')
torch.npu.set_device(int(NPU_ID))
DEVICE = f'npu:{torch.npu.current_device()}'

def cal_precision(input: torch.Tensor, golden: torch.Tensor):
    diff = torch.abs(input - golden)
    abs_max = diff.max().item()
    abs_mean = diff.mean().item()
    rmse = torch.sqrt(torch.mean((input - golden) ** 2)).item()
    mask = golden.abs() > 0.001
    if mask.any():
        rel_max = (diff[mask] / golden.abs()[mask]).max().item()
        rel_mean = (diff[mask] / golden.abs()[mask]).mean().item()
    else:
        rel_max = float('inf')
        rel_mean = float('inf')
    return (abs_max, abs_mean, rel_max, rel_mean, rmse)

def third_part_cmp(y_npu, y_gpu, ref, name):
    abs_max_npu, abs_mean_npu, rel_max_npu, rel_mean_npu, rmse_npu = cal_precision(y_npu.to('cpu').double(), ref)
    abs_max_gpu, abs_mean_gpu, rel_max_gpu, rel_mean_gpu, rmse_gpu = cal_precision(y_gpu.to('cpu').double(), ref)
    print(f'{name}:')
    print(f'  NPU: abs_max={abs_max_npu:.6f}, abs_mean={abs_mean_npu:.6f}, rel_max={rel_max_npu:.6f}, rel_mean={rel_mean_npu:.6f}, rmse={rmse_npu:.6f}')
    print(f'  GPU: abs_max={abs_max_gpu:.6f}, abs_mean={abs_mean_gpu:.6f}, rel_max={rel_max_gpu:.6f}, rel_mean={rel_mean_gpu:.6f}, rmse={rmse_gpu:.6f}')
    print(f'  NPU/GPU误差比: abs_max={abs_max_npu / abs_max_gpu:.4f}, rel_max={rel_max_npu / rel_max_gpu:.4f}, rmse={rmse_npu / rmse_gpu:.4f}')

def naive_fused_chunk_ttt_linear_fwd(q, k, v, w, b, eta, scale, eps, h0, hb0, output_final_state, BT, dtype=torch.float64):
    B, T, H, K = q.shape
    V = v.shape[-1]
    NT = math.ceil(T / BT)
    o = torch.zeros(B, T, H, V, dtype=dtype)
    ht = torch.zeros(B, H, K, V, dtype=dtype) if output_final_state else None
    hbt = torch.zeros(B, H, 1, V, dtype=dtype) if output_final_state else None
    for i_b in range(B):
        for i_h in range(H):
            b_h = h0[i_b, i_h].clone().to(dtype)
            b_hb = hb0[i_b, i_h].clone().to(dtype).squeeze(0)
            b_w = w[i_h].to(dtype)
            b_b = b[i_h].to(dtype)
            for i_t in range(NT):
                t_start = i_t * BT
                t_end = min((i_t + 1) * BT, T)
                cur_BT = t_end - t_start
                k_chunk = k[i_b, t_start:t_end, i_h].to(dtype)
                v_chunk = v[i_b, t_start:t_end, i_h].to(dtype)
                b_kh = k_chunk @ b_h + b_hb.unsqueeze(0)
                mean = b_kh.mean(dim=1, keepdim=True)
                xbar = b_kh - mean
                var = (xbar ** 2).mean(dim=1, keepdim=True)
                rstd = 1.0 / torch.sqrt(var + eps)
                b_kh_hat = (b_kh - mean) * rstd
                b_v_new = b_kh_hat * b_w.unsqueeze(0) + b_b.unsqueeze(0) - v_chunk + k_chunk
                b_v_new = b_v_new * b_w.unsqueeze(0)
                b_v2 = rstd * (V * b_v_new - b_v_new.sum(dim=1, keepdim=True) - b_kh_hat * (b_v_new * b_kh_hat).sum(dim=1, keepdim=True)) / V
                q_chunk = q[i_b, t_start:t_end, i_h].to(dtype)
                e_chunk = eta[i_b, t_start:t_end, i_h, 0].to(dtype)
                q_scaled = q_chunk * scale
                b_A = q_scaled @ k_chunk.T
                causal_mask = torch.tril(torch.ones(cur_BT, cur_BT, dtype=dtype))
                b_A = b_A * causal_mask
                b_Ae = e_chunk.unsqueeze(1) * causal_mask
                b_o = -(e_chunk.unsqueeze(1) * b_A) @ b_v2
                b_o = b_o + b_hb.unsqueeze(0) - b_Ae @ b_v2
                b_o = b_o + q_scaled @ b_h
                o[i_b, t_start:t_end, i_h] = b_o
                if i_t == NT - 1:
                    e_last = eta[i_b, T - 1, i_h, 0].to(dtype)
                else:
                    e_last = eta[i_b, i_t * BT + BT - 1, i_h, 0].to(dtype)
                b_h = b_h - e_last * k_chunk.T @ b_v2
                b_hb = b_hb - e_last * b_v2.sum(dim=0)
            if output_final_state:
                ht[i_b, i_h] = b_h
                hbt[i_b, i_h, 0, :] = b_hb
    return (o, ht, hbt)

@pytest.mark.functiontest
def test_fused_chunk_ttt_linear_fwd():
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/fused_chunk_ttt_linear_fwd_kernel'
    inputs = load_fixture(os.path.join(load_dir, 'inputs.pt'), map_location=torch.device('cpu'))
    q = inputs['q']
    k = inputs['k']
    v = inputs['v']
    w = inputs['w']
    b = inputs['b']
    eta = inputs['eta']
    sc = inputs['sc']
    eps_val = inputs['eps_val']
    h0 = inputs['h0']
    hb0 = inputs['hb0']
    out_final = inputs['out_final']
    cu_sl = inputs['cu_sl']
    bt = inputs['bt']
    cpu_o, cpu_ht, cpu_hbt = naive_fused_chunk_ttt_linear_fwd(q.detach().clone().to(torch.float64), k.detach().clone().to(torch.float64), v.detach().clone().to(torch.float64), w.detach().clone().to(torch.float64), b.detach().clone().to(torch.float64), eta.detach().clone().to(torch.float64), sc, eps_val, h0.detach().clone().to(torch.float64), hb0.detach().clone().to(torch.float64), out_final, bt, dtype=torch.float64)
    from yb_14d6a471f4f932 import fused_chunk_ttt_linear_fwd
    q_npu = q.to(DEVICE)
    k_npu = k.to(DEVICE)
    v_npu = v.to(DEVICE)
    w_npu = w.to(DEVICE)
    b_npu = b.to(DEVICE)
    eta_npu = eta.to(DEVICE)
    h0_npu = h0.to(DEVICE)
    hb0_npu = hb0.to(DEVICE)
    npu_o, npu_ht, npu_hbt = fused_chunk_ttt_linear_fwd(q_npu, k_npu, v_npu, w_npu, b_npu, eta_npu, sc, eps_val, h0_npu, hb0_npu, out_final, cu_sl, bt)
    torch.npu.synchronize()
    expected_outputs = load_fixture(os.path.join(load_dir, 'outputs.pt'), map_location=torch.device('cpu'))
    gpu_o = expected_outputs['output_0']
    gpu_ht = expected_outputs['output_1']
    gpu_hbt = expected_outputs['output_2']
    print('=' * 70)
    print('三方对比: NPU vs GPU (均以 CPU FP64 为参考)')
    print('=' * 70)
    third_part_cmp(npu_o, gpu_o, cpu_o, 'o')
    if out_final:
        third_part_cmp(npu_ht, gpu_ht, cpu_ht, 'ht')
        third_part_cmp(npu_hbt, gpu_hbt, cpu_hbt, 'hbt')
    print('\n' + '=' * 70)
    print('NPU vs CPU FP64 直接对比')
    print('=' * 70)
    atol = 0.01
    rtol = 0.01
    for name, npu_val, cpu_val in [('o', npu_o, cpu_o)] + ([('ht', npu_ht, cpu_ht), ('hbt', npu_hbt, cpu_hbt)] if out_final else []):
        npu_cpu = npu_val.to('cpu').double()
        max_diff = (npu_cpu - cpu_val).abs().max().item()
        mean_diff = (npu_cpu - cpu_val).abs().mean().item()
        if torch.allclose(npu_cpu.float(), cpu_val.float(), atol=atol, rtol=rtol):
            print(f'✅ [{name}] NPU and CPU FP64 ref match (max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f})')
        else:
            print(f'❌ [{name}] NPU and CPU FP64 ref differ (max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f})')
    print('\n' + '=' * 70)
    print('GPU vs CPU FP64 直接对比 (GPU精度基线)')
    print('=' * 70)
    for name, gpu_val, cpu_val in [('o', gpu_o, cpu_o)] + ([('ht', gpu_ht, cpu_ht), ('hbt', gpu_hbt, cpu_hbt)] if out_final else []):
        gpu_double = gpu_val.double()
        max_diff = (gpu_double - cpu_val).abs().max().item()
        mean_diff = (gpu_double - cpu_val).abs().mean().item()
        if torch.allclose(gpu_double.float(), cpu_val.float(), atol=atol, rtol=rtol):
            print(f'✅ [{name}] GPU and CPU FP64 ref match (max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f})')
        else:
            print(f'❌ [{name}] GPU and CPU FP64 ref differ (max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f})')
    assert torch.allclose(npu_o.to('cpu').double().float(), cpu_o.float(), atol=atol, rtol=rtol), f"[o] NPU and CPU FP64 ref differ, max_diff={(npu_o.to('cpu').double() - cpu_o).abs().max():.6f}"
    if out_final:
        assert torch.allclose(npu_ht.to('cpu').double().float(), cpu_ht.float(), atol=atol, rtol=rtol), f"[ht] NPU and CPU FP64 ref differ, max_diff={(npu_ht.to('cpu').double() - cpu_ht).abs().max():.6f}"
        assert torch.allclose(npu_hbt.to('cpu').double().float(), cpu_hbt.float(), atol=atol, rtol=rtol), f"[hbt] NPU and CPU FP64 ref differ, max_diff={(npu_hbt.to('cpu').double() - cpu_hbt).abs().max():.6f}"
