
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
    if abs_max_gpu > 0:
        print(f'  NPU/GPU误差比: abs_max={abs_max_npu / abs_max_gpu:.4f}, rel_max={rel_max_npu / rel_max_gpu:.4f}, rmse={rmse_npu / rmse_gpu:.4f}')
    else:
        print(f'  NPU/GPU误差比: GPU abs_max=0, 无法计算比值')

def naive_chunk_bwd_kernel_dqke(q, k, v, eta, h, do, dh, dhb, scale, chunk_size, dtype=torch.float64):
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    NT = math.ceil(T / BT)
    q = q.to(dtype)
    k = k.to(dtype)
    v = v.to(dtype)
    eta = eta.to(dtype)
    h = h.to(dtype)
    do = do.to(dtype)
    dh = dh.to(dtype)
    dhb = dhb.to(dtype)
    dq = torch.zeros(B, T, H, K, dtype=dtype)
    dk = torch.zeros(B, T, H, K, dtype=dtype)
    de = torch.zeros(B, T, H, 1, dtype=dtype)
    for i_b in range(B):
        for i_h in range(H):
            for i_t in range(NT):
                t_start = i_t * BT
                t_end = min((i_t + 1) * BT, T)
                cur_BT = t_end - t_start
                q_c = q[i_b, t_start:t_end, i_h, :]
                k_c = k[i_b, t_start:t_end, i_h, :]
                v_c = v[i_b, t_start:t_end, i_h, :]
                e_c = eta[i_b, t_start:t_end, i_h, 0]
                h_c = h[i_b, i_t, i_h, :, :]
                do_c = do[i_b, t_start:t_end, i_h, :]
                dh_c = dh[i_b, i_t, i_h, :, :]
                dhb_c = dhb[i_b, i_t, i_h, 0, :]
                e_last = e_c[-1]
                ds = do_c @ v_c.T
                causal_mask = torch.tril(torch.ones(cur_BT, cur_BT, dtype=dtype))
                ds = ds * causal_mask
                dq_c = (do_c @ h_c.T - ds @ k_c * e_c[:, None]) * scale
                dk_c = -e_last * (v_c @ dh_c.T) - ds.T @ (q_c * e_c[:, None]) * scale
                de_c = torch.zeros(cur_BT, dtype=dtype)
                de_c[-1] -= torch.sum(dh_c * (k_c.T @ v_c))
                de_c[-1] -= torch.sum(dhb_c * v_c.sum(dim=0))
                de_c -= torch.sum(scale * ds @ k_c * q_c, dim=1)
                de_c -= torch.sum(ds, dim=1)
                dq[i_b, t_start:t_end, i_h, :] = dq_c
                dk[i_b, t_start:t_end, i_h, :] = dk_c
                de[i_b, t_start:t_end, i_h, 0] = de_c
    return (dq, dk, de)

@pytest.mark.functiontest
def test_chunk_bwd_kernel_dqke():
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_bwd_kernel_dqke'
    q = load_fixture(os.path.join(load_dir, 'q.pt'), map_location=torch.device('cpu'))
    k = load_fixture(os.path.join(load_dir, 'k.pt'), map_location=torch.device('cpu'))
    v_new = load_fixture(os.path.join(load_dir, 'v_new.pt'), map_location=torch.device('cpu'))
    eta = load_fixture(os.path.join(load_dir, 'eta.pt'), map_location=torch.device('cpu'))
    h = load_fixture(os.path.join(load_dir, 'h.pt'), map_location=torch.device('cpu'))
    do_tensor = load_fixture(os.path.join(load_dir, 'do_tensor.pt'), map_location=torch.device('cpu'))
    dh = load_fixture(os.path.join(load_dir, 'dh.pt'), map_location=torch.device('cpu'))
    dhb = load_fixture(os.path.join(load_dir, 'dhb.pt'), map_location=torch.device('cpu'))
    sc = load_fixture(os.path.join(load_dir, 'scale.pt')).item()
    cs = load_fixture(os.path.join(load_dir, 'chunk_size.pt')).item()
    cpu_dq, cpu_dk, cpu_de = naive_chunk_bwd_kernel_dqke(q.detach().clone(), k.detach().clone(), v_new.detach().clone(), eta.detach().clone(), h.detach().clone(), do_tensor.detach().clone(), dh.detach().clone(), dhb.detach().clone(), sc, cs, dtype=torch.float64)
    from yb_7bff0dd87eff27 import chunk_ttt_linear_bwd_dqke
    q_npu = q.to(DEVICE)
    k_npu = k.to(DEVICE)
    v_new_npu = v_new.to(DEVICE)
    eta_npu = eta.to(DEVICE)
    h_npu = h.to(DEVICE)
    do_npu = do_tensor.to(DEVICE)
    dh_npu = dh.to(DEVICE)
    dhb_npu = dhb.to(DEVICE)
    npu_dq, npu_dk, npu_de = chunk_ttt_linear_bwd_dqke(q_npu, k_npu, v_new_npu, eta_npu, h_npu, do_npu, dh_npu, dhb_npu, sc, None, cs)
    torch.npu.synchronize()
    gpu_dq = load_fixture(os.path.join(load_dir, 'output_0.pt'), map_location=torch.device('cpu'))
    gpu_dk = load_fixture(os.path.join(load_dir, 'output_1.pt'), map_location=torch.device('cpu'))
    gpu_de = load_fixture(os.path.join(load_dir, 'output_2.pt'), map_location=torch.device('cpu'))
    print('=' * 70)
    print('三方对比: NPU FP16 vs GPU FP16 (均以 CPU FP64 为参考)')
    print('=' * 70)
    third_part_cmp(npu_dq, gpu_dq, cpu_dq, 'dq')
    third_part_cmp(npu_dk, gpu_dk, cpu_dk, 'dk')
    third_part_cmp(npu_de, gpu_de, cpu_de, 'de')
    atol = 0.05
    rtol = 0.05
    output_configs = [('dq', npu_dq, cpu_dq), ('dk', npu_dk, cpu_dk), ('de', npu_de, cpu_de)]
    print('\n' + '=' * 70)
    print('NPU vs CPU FP64 直接对比')
    print('=' * 70)
    for name, npu_val, cpu_val in output_configs:
        npu_cpu = npu_val.to('cpu').double()
        max_diff = (npu_cpu - cpu_val).abs().max().item()
        mean_diff = (npu_cpu - cpu_val).abs().mean().item()
        if torch.allclose(npu_cpu.float(), cpu_val.float(), atol=atol, rtol=rtol):
            print(f'  [{name}] NPU and CPU FP64 ref match (atol={atol:.4f}, max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f})')
        else:
            print(f'  [{name}] NPU and CPU FP64 ref differ (atol={atol:.4f}, max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f})')
    print('\n' + '=' * 70)
    print('GPU vs CPU FP64 直接对比 (GPU精度基线)')
    print('=' * 70)
    for name, gpu_val, cpu_val in [('dq', gpu_dq, cpu_dq), ('dk', gpu_dk, cpu_dk), ('de', gpu_de, cpu_de)]:
        gpu_double = gpu_val.double()
        max_diff = (gpu_double - cpu_val).abs().max().item()
        mean_diff = (gpu_double - cpu_val).abs().mean().item()
        if torch.allclose(gpu_double.float(), cpu_val.float(), atol=atol, rtol=rtol):
            print(f'  [{name}] GPU and CPU FP64 ref match (atol={atol:.4f}, max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f})')
        else:
            print(f'  [{name}] GPU and CPU FP64 ref differ (atol={atol:.4f}, max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f})')
    print('\n' + '=' * 70)
    print('断言: NPU 与 CPU FP64 参考在容差范围内一致')
    print('=' * 70)
    for name, npu_val, cpu_val in output_configs:
        npu_cpu = npu_val.to('cpu').double().float()
        assert torch.allclose(npu_cpu, cpu_val.float(), atol=atol, rtol=rtol), f"[{name}] NPU and CPU FP64 ref differ, max_diff={(npu_val.to('cpu').double() - cpu_val).abs().max():.6f}, atol={atol:.4f}"
        print(f'  [{name}] assertion passed (atol={atol:.4f})')
    print('All assertions passed!')
