
from opbench.fla_fixtures import load_fixture
import os
import pytest
import math
import torch
import torch_npu
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

def naive_chunk_oja_bwd_dhu(q, vg, w, do, dk, gv, h0, dht, scale, chunk_size, dtype=torch.float64):
    B, T, H, K = q.shape
    V = vg.shape[-1]
    BT = 64
    NT = math.ceil(T / BT)
    dh = torch.zeros(B, NT, H, K, V, dtype=dtype)
    dh0 = torch.zeros(B, H, K, V, dtype=dtype)
    dk2 = torch.zeros(B, T, H, K, dtype=dtype)
    for i_b in range(B):
        for i_h in range(H):
            b_dh = dht[i_b, i_h].clone()
            for i_t in range(NT - 1, -1, -1):
                t_start = i_t * BT
                t_end = min((i_t + 1) * BT, T)
                last_idx = t_end - 1
                dh[i_b, i_t, i_h] = b_dh
                vg_chunk = vg[i_b, t_start:t_end, i_h]
                dk_dot = vg_chunk @ b_dh.T
                b_dk2 = dk_dot + dk[i_b, t_start:t_end, i_h]
                dk2[i_b, t_start:t_end, i_h] = b_dk2
                gv_last = gv[i_b, last_idx, i_h]
                b_dh = b_dh * torch.exp(gv_last)[None, :]
                q_chunk = q[i_b, t_start:t_end, i_h]
                do_chunk = do[i_b, t_start:t_end, i_h]
                gv_chunk = gv[i_b, t_start:t_end, i_h]
                w_chunk = w[i_b, t_start:t_end, i_h]
                do_gated = do_chunk * torch.exp(gv_chunk)
                qdo = q_chunk.T @ do_gated * scale
                dkw = b_dk2.T @ w_chunk
                b_dh = b_dh + qdo - dkw
            dh0[i_b, i_h] = b_dh
    return (dh, dh0, dk2)

@pytest.mark.functiontest
def test_oja_bwd_kernel_dhu_function():
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_oja_bwd_kernel_dhu_blockdim64'
    q = load_fixture(os.path.join(load_dir, 'q.pt'), map_location=torch.device('cpu'))
    vg = load_fixture(os.path.join(load_dir, 'vg.pt'), map_location=torch.device('cpu'))
    w = load_fixture(os.path.join(load_dir, 'w.pt'), map_location=torch.device('cpu'))
    do = load_fixture(os.path.join(load_dir, 'do.pt'), map_location=torch.device('cpu'))
    dk = load_fixture(os.path.join(load_dir, 'dk.pt'), map_location=torch.device('cpu'))
    gv = load_fixture(os.path.join(load_dir, 'gv.pt'), map_location=torch.device('cpu'))
    h0 = load_fixture(os.path.join(load_dir, 'h0.pt'), map_location=torch.device('cpu'))
    dht = load_fixture(os.path.join(load_dir, 'dht.pt'), map_location=torch.device('cpu'))
    scale = load_fixture(os.path.join(load_dir, 'scale.pt')).item()
    chunk_size = load_fixture(os.path.join(load_dir, 'chunk_size.pt')).item()
    q_cpu = q.detach().clone().to(torch.float64)
    vg_cpu = vg.detach().clone().to(torch.float64)
    w_cpu = w.detach().clone().to(torch.float64)
    do_cpu = do.detach().clone().to(torch.float64)
    dk_cpu = dk.detach().clone().to(torch.float64)
    gv_cpu = gv.detach().clone().to(torch.float64)
    h0_cpu = h0.detach().clone().to(torch.float64)
    dht_cpu = dht.detach().clone().to(torch.float64)
    cpu_dh, cpu_dh0, cpu_dk2 = naive_chunk_oja_bwd_dhu(q_cpu, vg_cpu, w_cpu, do_cpu, dk_cpu, gv_cpu, h0_cpu, dht_cpu, scale, chunk_size, dtype=torch.float64)
    from yb_5a9aa3bd6a5163 import chunk_oja_bwd_dhu
    q_npu = q.to(DEVICE)
    vg_npu = vg.to(DEVICE)
    w_npu = w.to(DEVICE)
    do_npu = do.to(DEVICE)
    dk_npu = dk.to(DEVICE)
    gv_npu = gv.to(DEVICE)
    h0_npu = h0.to(DEVICE)
    dht_npu = dht.to(DEVICE)
    npu_dh, npu_dh0, npu_dk2 = chunk_oja_bwd_dhu(q_npu, vg_npu, w_npu, do_npu, dk_npu, gv=gv_npu, h0=h0_npu, dht=dht_npu, scale=scale, chunk_size=chunk_size)
    torch.npu.synchronize()
    gpu_dh = load_fixture(os.path.join(load_dir, 'output_0.pt'), map_location=torch.device('cpu'))
    gpu_dh0 = load_fixture(os.path.join(load_dir, 'output_1.pt'), map_location=torch.device('cpu'))
    gpu_dk2 = load_fixture(os.path.join(load_dir, 'output_2.pt'), map_location=torch.device('cpu'))
    print('=' * 70)
    print('三方对比: NPU vs GPU (均以 CPU FP64 为参考)')
    print('=' * 70)
    third_part_cmp(npu_dh, gpu_dh, cpu_dh, 'dh')
    third_part_cmp(npu_dh0, gpu_dh0, cpu_dh0, 'dh0')
    third_part_cmp(npu_dk2, gpu_dk2, cpu_dk2, 'dk2')
    print('\n' + '=' * 70)
    print('NPU vs CPU FP64 直接对比')
    print('=' * 70)
    atol = 0.01
    rtol = 0.01
    for name, npu_val, cpu_val in [('dh', npu_dh, cpu_dh), ('dh0', npu_dh0, cpu_dh0), ('dk2', npu_dk2, cpu_dk2)]:
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
    for name, gpu_val, cpu_val in [('dh', gpu_dh, cpu_dh), ('dh0', gpu_dh0, cpu_dh0), ('dk2', gpu_dk2, cpu_dk2)]:
        gpu_double = gpu_val.double()
        max_diff = (gpu_double - cpu_val).abs().max().item()
        mean_diff = (gpu_double - cpu_val).abs().mean().item()
        if torch.allclose(gpu_double.float(), cpu_val.float(), atol=atol, rtol=rtol):
            print(f'✅ [{name}] GPU and CPU FP64 ref match (max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f})')
        else:
            print(f'❌ [{name}] GPU and CPU FP64 ref differ (max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f})')
    assert torch.allclose(npu_dh.to('cpu').double().float(), cpu_dh.float(), atol=atol, rtol=rtol), f"[dh] NPU and CPU FP64 ref differ, max_diff={(npu_dh.to('cpu').double() - cpu_dh).abs().max():.6f}"
    assert torch.allclose(npu_dh0.to('cpu').double().float(), cpu_dh0.float(), atol=atol, rtol=rtol), f"[dh0] NPU and CPU FP64 ref differ, max_diff={(npu_dh0.to('cpu').double() - cpu_dh0).abs().max():.6f}"
    assert torch.allclose(npu_dk2.to('cpu').double().float(), cpu_dk2.float(), atol=atol, rtol=rtol), f"[dk2] NPU and CPU FP64 ref differ, max_diff={(npu_dk2.to('cpu').double() - cpu_dk2).abs().max():.6f}"
