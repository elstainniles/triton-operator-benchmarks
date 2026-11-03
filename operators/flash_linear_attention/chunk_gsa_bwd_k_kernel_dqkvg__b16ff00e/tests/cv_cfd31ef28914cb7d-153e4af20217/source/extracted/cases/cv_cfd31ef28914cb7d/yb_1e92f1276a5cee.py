
from opbench.fla_fixtures import load_fixture
import os
import pytest
import math
import torch
import torch_npu
import triton
NPU_ID = os.environ.get('BENCH_NPU_ID', '0')
torch.npu.set_device(int(NPU_ID))
DEVICE = f'npu:{torch.npu.current_device()}'

def cal_precision(input: torch.Tensor, golden: torch.Tensor):
    relative_max = torch.max(torch.abs(input - golden) / (torch.abs(golden) + float(1e-07)))
    relative_mean = torch.mean(torch.abs(input - golden) / (torch.abs(golden) + float(1e-07)))
    rmse = torch.sqrt(torch.mean((input - golden) ** 2))
    return (relative_max, relative_mean, rmse)

def third_part_cmp(y_npu, y_gpu, ref, name):
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(y_npu.to('cpu'), ref)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(y_gpu.to('cpu'), ref)
    print(f'{name} 最大相对误差比 :{relative_max_npu / relative_max_gpu}, 平均相对误差比 :{relative_mean_npu / relative_mean_gpu}, 均方根误差比 :{rmse_npu / rmse_gpu}')

def naive_chunk_gsa_bwd_k_dqkvg(q, k, v, g, h, do, dh, dA, dg, scale, BT, dtype=torch.float64):
    B, T, HQ, K = q.shape
    H = k.shape[2]
    V = v.shape[-1]
    NT = math.ceil(T / BT)
    NG = HQ // H
    A_out = torch.zeros(B, T, HQ, BT, dtype=dtype)
    dq_out = torch.zeros(B, T, HQ, K, dtype=dtype)
    dk_out = torch.zeros(B, T, HQ, K, dtype=dtype)
    dv_out = torch.zeros(B, T, HQ, V, dtype=dtype)
    dgv_out = torch.zeros(B, T, HQ, V, dtype=dtype)
    for i_b in range(B):
        for i_t in range(NT):
            t_start = i_t * BT
            t_end = min(t_start + BT, T)
            actual_bt = t_end - t_start
            o_t = t_end
            for i_hq in range(HQ):
                i_h = i_hq // NG
                q_block = q[i_b, t_start:t_end, i_hq, :]
                k_block = k[i_b, t_start:t_end, i_h, :]
                v_block = v[i_b, t_start:t_end, i_h, :]
                g_block = g[i_b, t_start:t_end, i_h, :]
                do_block = do[i_b, t_start:t_end, i_hq, :]
                dA_block = dA[i_b, t_start:t_end, i_hq, :actual_bt]
                dg_block = dg[i_b, t_start:t_end, i_hq, :]
                h_block = h[i_b, i_t, i_h, :, :]
                dh_block = dh[i_b, i_t, i_hq, :, :]
                g_last = g[i_b, o_t - 1, i_h, :]
                gv = torch.exp2(g_last.unsqueeze(0) - g_block)
                A_block = torch.mm(q_block * scale, k_block.T)
                causal_mask = torch.tril(torch.ones(actual_bt, actual_bt, dtype=dtype))
                A_block = A_block * causal_mask
                A_out[i_b, t_start:t_end, i_hq, :actual_bt] = A_block
                do_scaled = do_block * torch.exp2(g_block)
                dg_scalar = torch.sum(h_block * dh_block, dim=0) * torch.exp2(g_last)
                dq_chunk = torch.mm(do_scaled, h_block.T) * scale
                dk_chunk = torch.mm(v_block * gv, dh_block.T)
                dv_chunk = torch.mm(k_block, dh_block) * gv
                dg_scalar = dg_scalar + torch.sum(dv_chunk * v_block, dim=0)
                dq_chunk = dq_chunk + torch.mm(dA_block, k_block)
                dk_chunk = dk_chunk + torch.mm(dA_block.T, q_block)
                dgv_chunk = dg_block + dg_scalar.unsqueeze(0)
                dq_out[i_b, t_start:t_end, i_hq, :] = dq_chunk
                dk_out[i_b, t_start:t_end, i_hq, :] = dk_chunk
                dv_out[i_b, t_start:t_end, i_hq, :] = dv_chunk
                dgv_out[i_b, t_start:t_end, i_hq, :] = dgv_chunk
    return (A_out, dq_out, dk_out, dv_out, dgv_out)

@pytest.mark.functiontest
def test_chunk_gsa_bwd_k_kernel_dqkvg_function():
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_gsa_bwd_k_kernel_dqkvg'
    q = load_fixture(os.path.join(load_dir, 'q.pt'), map_location=torch.device('cpu'))
    k = load_fixture(os.path.join(load_dir, 'k.pt'), map_location=torch.device('cpu'))
    v = load_fixture(os.path.join(load_dir, 'v.pt'), map_location=torch.device('cpu'))
    g = load_fixture(os.path.join(load_dir, 'g.pt'), map_location=torch.device('cpu'))
    h = load_fixture(os.path.join(load_dir, 'h.pt'), map_location=torch.device('cpu'))
    do = load_fixture(os.path.join(load_dir, 'do.pt'), map_location=torch.device('cpu'))
    dh = load_fixture(os.path.join(load_dir, 'dh.pt'), map_location=torch.device('cpu'))
    dA = load_fixture(os.path.join(load_dir, 'dA.pt'), map_location=torch.device('cpu'))
    dg = load_fixture(os.path.join(load_dir, 'dg.pt'), map_location=torch.device('cpu'))
    scale = load_fixture(os.path.join(load_dir, 'scale.pt')).item()
    BT = load_fixture(os.path.join(load_dir, 'chunk_size.pt')).item()
    q_cpu = q.detach().clone().to(torch.float64)
    k_cpu = k.detach().clone().to(torch.float64)
    v_cpu = v.detach().clone().to(torch.float64)
    g_cpu = g.detach().clone().to(torch.float64)
    h_cpu = h.detach().clone().to(torch.float64)
    do_cpu = do.detach().clone().to(torch.float64)
    dh_cpu = dh.detach().clone().to(torch.float64)
    dA_cpu = dA.detach().clone().to(torch.float64)
    dg_cpu = dg.detach().clone().to(torch.float64)
    cpu_A, cpu_dq, cpu_dk, cpu_dv, cpu_dgv = naive_chunk_gsa_bwd_k_dqkvg(q_cpu, k_cpu, v_cpu, g_cpu, h_cpu, do_cpu, dh_cpu, dA_cpu, dg_cpu, scale, BT, dtype=torch.float64)
    from yb_b16ff00ec8999a import chunk_gsa_bwd_k_kernel_dqkvg
    q_npu = q.to(DEVICE)
    k_npu = k.to(DEVICE)
    v_npu = v.to(DEVICE)
    g_npu = g.to(DEVICE)
    h_npu = h.to(DEVICE)
    do_npu = do.to(DEVICE)
    dh_npu = dh.to(DEVICE)
    dA_npu = dA.to(DEVICE)
    dg_npu = dg.to(DEVICE)
    B_val, T_val, HQ_val, K_val = q_npu.shape
    H_val = k_npu.shape[2]
    V_val = v_npu.shape[-1]
    BK = min(64, triton.next_power_of_2(K_val))
    BV = min(64, triton.next_power_of_2(V_val))
    NK = triton.cdiv(K_val, BK)
    NG = HQ_val // H_val
    NT = triton.cdiv(T_val, BT)
    A_npu = torch.empty(NK, B_val, T_val, HQ_val, BT, dtype=q_npu.dtype, device=DEVICE)
    dq_npu = torch.empty(B_val, T_val, HQ_val, K_val, dtype=q_npu.dtype, device=DEVICE)
    dk_npu = torch.empty(B_val, T_val, HQ_val, K_val, dtype=q_npu.dtype, device=DEVICE)
    dv_npu = torch.empty(NK, B_val, T_val, HQ_val, V_val, dtype=q_npu.dtype, device=DEVICE)
    dgv_npu = torch.empty(NK, B_val, T_val, HQ_val, V_val, dtype=torch.float32, device=DEVICE)
    grid = (NK, NT, B_val * HQ_val)
    chunk_gsa_bwd_k_kernel_dqkvg[grid](q_npu, k_npu, v_npu, h_npu, g_npu, A_npu, do_npu, dh_npu, dq_npu, dk_npu, dv_npu, dg_npu, dgv_npu, dA_npu, cu_seqlens=None, chunk_indices=None, scale=scale, T=T_val, B=B_val, HQ=HQ_val, H=H_val, K=K_val, V=V_val, BT=BT, BK=BK, BV=BV, NG=NG)
    torch.npu.synchronize()
    npu_A = A_npu.sum(0)
    npu_dq = dq_npu
    npu_dk = dk_npu
    npu_dv = dv_npu.sum(0)
    npu_dgv = dgv_npu.sum(0)
    gpu_A = load_fixture(os.path.join(load_dir, 'output_0.pt'), map_location=torch.device('cpu'))
    gpu_dq = load_fixture(os.path.join(load_dir, 'output_1.pt'), map_location=torch.device('cpu'))
    gpu_dk = load_fixture(os.path.join(load_dir, 'output_2.pt'), map_location=torch.device('cpu'))
    gpu_dv = load_fixture(os.path.join(load_dir, 'output_3.pt'), map_location=torch.device('cpu'))
    gpu_dgv = load_fixture(os.path.join(load_dir, 'output_4.pt'), map_location=torch.device('cpu'))
    third_part_cmp(npu_A, gpu_A, cpu_A, 'A')
    third_part_cmp(npu_dq, gpu_dq, cpu_dq, 'dq')
    third_part_cmp(npu_dk, gpu_dk, cpu_dk, 'dk')
    third_part_cmp(npu_dv, gpu_dv, cpu_dv, 'dv')
    third_part_cmp(npu_dgv, gpu_dgv, cpu_dgv, 'dgv')
    atol = 0.01
    rtol = 0.01
    if torch.allclose(npu_A.to('cpu'), cpu_A.float(), atol=atol, rtol=rtol):
        print('  [A] NPU FP32 and CPU FP64 ref match')
    else:
        print('  [A] NPU FP32 and CPU FP64 ref differ')
    if torch.allclose(npu_dq.to('cpu'), cpu_dq.float(), atol=atol, rtol=rtol):
        print('  [dq] NPU FP32 and CPU FP64 ref match')
    else:
        print('  [dq] NPU FP32 and CPU FP64 ref differ')
    if torch.allclose(npu_dk.to('cpu'), cpu_dk.float(), atol=atol, rtol=rtol):
        print('  [dk] NPU FP32 and CPU FP64 ref match')
    else:
        print('  [dk] NPU FP32 and CPU FP64 ref differ')
    if torch.allclose(npu_dv.to('cpu'), cpu_dv.float(), atol=atol, rtol=rtol):
        print('  [dv] NPU FP32 and CPU FP64 ref match')
    else:
        print('  [dv] NPU FP32 and CPU FP64 ref differ')
    if torch.allclose(npu_dgv.to('cpu'), cpu_dgv.float(), atol=atol, rtol=rtol):
        print('  [dgv] NPU FP32 and CPU FP64 ref match')
    else:
        print('  [dgv] NPU FP32 and CPU FP64 ref differ')
    if torch.allclose(gpu_A, cpu_A.float(), atol=atol, rtol=rtol):
        print('  [A] GPU FP32 and CPU FP64 ref match')
    else:
        print('  [A] GPU FP32 and CPU FP64 ref differ')
    if torch.allclose(gpu_dq, cpu_dq.float(), atol=atol, rtol=rtol):
        print('  [dq] GPU FP32 and CPU FP64 ref match')
    else:
        print('  [dq] GPU FP32 and CPU FP64 ref differ')
    if torch.allclose(gpu_dk, cpu_dk.float(), atol=atol, rtol=rtol):
        print('  [dk] GPU FP32 and CPU FP64 ref match')
    else:
        print('  [dk] GPU FP32 and CPU FP64 ref differ')
    if torch.allclose(gpu_dv, cpu_dv.float(), atol=atol, rtol=rtol):
        print('  [dv] GPU FP32 and CPU FP64 ref match')
    else:
        print('  [dv] GPU FP32 and CPU FP64 ref differ')
    if torch.allclose(gpu_dgv, cpu_dgv.float(), atol=atol, rtol=rtol):
        print('  [dgv] GPU FP32 and CPU FP64 ref match')
    else:
        print('  [dgv] GPU FP32 and CPU FP64 ref differ')
    assert torch.allclose(npu_A.to('cpu'), cpu_A.float(), atol=atol, rtol=rtol), f"[A] NPU FP32 and CPU FP64 ref differ, max_diff={(npu_A.to('cpu').double() - cpu_A).abs().max():.6f}"
    assert torch.allclose(npu_dq.to('cpu'), cpu_dq.float(), atol=atol, rtol=rtol), f"[dq] NPU FP32 and CPU FP64 ref differ, max_diff={(npu_dq.to('cpu').double() - cpu_dq).abs().max():.6f}"
    assert torch.allclose(npu_dk.to('cpu'), cpu_dk.float(), atol=atol, rtol=rtol), f"[dk] NPU FP32 and CPU FP64 ref differ, max_diff={(npu_dk.to('cpu').double() - cpu_dk).abs().max():.6f}"
    assert torch.allclose(npu_dv.to('cpu'), cpu_dv.float(), atol=atol, rtol=rtol), f"[dv] NPU FP32 and CPU FP64 ref differ, max_diff={(npu_dv.to('cpu').double() - cpu_dv).abs().max():.6f}"
    assert torch.allclose(npu_dgv.to('cpu'), cpu_dgv.float(), atol=atol, rtol=rtol), f"[dgv] NPU FP32 and CPU FP64 ref differ, max_diff={(npu_dgv.to('cpu').double() - cpu_dgv).abs().max():.6f}"
