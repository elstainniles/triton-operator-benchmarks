
from opbench.fla_fixtures import load_fixture
import os
import math
import torch
import torch_npu
import pytest
NPU_ID = os.environ.get('BENCH_NPU_ID', '0')
torch.npu.set_device(int(NPU_ID))
DEVICE = f'npu:{torch.npu.current_device()}'

def cal_precision(input: torch.tensor, golden: torch.tensor):
    relative_max = torch.max(torch.abs(input - golden) / (torch.abs(golden) + float(1e-07)))
    print(f'  max relative error: {relative_max:.6e}')
    relative_mean = torch.mean(torch.abs(input - golden) / (torch.abs(golden) + float(1e-07)))
    print(f'  mean relative error: {relative_mean:.6e}')
    rmse = torch.sqrt(torch.mean((input - golden) ** 2))
    print(f'  rmse: {rmse:.6e}')
    mask = (torch.abs(golden) < float(2e-14)) & (torch.abs(input - golden) > float(2e-30))
    error = torch.abs(input - golden)
    total_error = torch.sum(error[mask])
    print(f'  small val error: {total_error:.6e}')
    return (relative_max, relative_mean, rmse)

def naive_chunk_ttt_linear_bwd_h(k, v, w, b, eta, eps, h0, hb0, BT, dtype=torch.float64):
    B, T, H, K = k.shape
    V = v.shape[-1]
    NT = math.ceil(T / BT)
    h_out = torch.zeros(B, NT, H, K, V, dtype=dtype)
    v_new_out = torch.zeros(B, T, H, V, dtype=dtype)
    x_out = torch.zeros(B, T, H, V, dtype=dtype)
    y_out = torch.zeros(B, T, H, V, dtype=dtype)
    rstd_out = torch.zeros(B, T, H, 1, dtype=dtype)
    for i_b in range(B):
        for i_h in range(H):
            b_h = h0[i_b, i_h].clone().to(dtype) if h0 is not None else torch.zeros(K, V, dtype=dtype)
            b_hb = hb0[i_b, i_h].clone().to(dtype).squeeze(0) if hb0 is not None else torch.zeros(V, dtype=dtype)
            b_w = w[i_h].to(dtype)
            b_b = b[i_h].to(dtype)
            for i_t in range(NT):
                h_out[i_b, i_t, i_h] = b_h
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
                b_v_new_raw = (b_kh_hat * b_w.unsqueeze(0) + b_b.unsqueeze(0) - v_chunk + k_chunk) * b_w.unsqueeze(0)
                v_mean = b_v_new_raw.mean(dim=1, keepdim=True)
                v_khhat_dot = (b_v_new_raw * b_kh_hat).mean(dim=1, keepdim=True)
                b_v2 = rstd * (b_v_new_raw - v_mean - b_kh_hat * v_khhat_dot)
                x_out[i_b, t_start:t_end, i_h] = b_kh_hat
                y_out[i_b, t_start:t_end, i_h] = b_v_new_raw
                rstd_out[i_b, t_start:t_end, i_h, 0] = rstd.squeeze(-1)
                v_new_out[i_b, t_start:t_end, i_h] = b_v2
                if i_t == NT - 1:
                    e_last = eta[i_b, T - 1, i_h, 0].to(dtype)
                else:
                    e_last = eta[i_b, i_t * BT + BT - 1, i_h, 0].to(dtype)
                b_h = b_h - e_last * k_chunk.T @ b_v2
                b_hb = b_hb - e_last * b_v2.sum(dim=0)
    return (h_out, v_new_out, x_out, y_out, rstd_out)

@pytest.mark.functiontest
def test_chunk_ttt_linear_bwd_kernel_h():
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_ttt_linear_bwd_kernel_h'
    inputs = load_fixture(os.path.join(load_dir, 'inputs.pt'), map_location=torch.device('cpu'))
    k = inputs['k']
    v = inputs['v']
    w = inputs['w']
    b = inputs['b']
    eta = inputs['eta']
    eps_val = inputs['eps_val']
    h0 = inputs['h0']
    hb0 = inputs['hb0']
    bt = inputs['chunk_size']
    cpu_h, cpu_v_new, cpu_x, cpu_y, cpu_rstd = naive_chunk_ttt_linear_bwd_h(k.detach().clone().to(torch.float64), v.detach().clone().to(torch.float64), w.detach().clone().to(torch.float64), b.detach().clone().to(torch.float64), eta.detach().clone().to(torch.float64), eps_val, h0.detach().clone().to(torch.float64), hb0.detach().clone().to(torch.float64), bt, dtype=torch.float64)
    from yb_7bff0dd87eff27 import chunk_ttt_linear_bwd_h
    k_npu = k.to(DEVICE)
    v_npu = v.to(DEVICE)
    w_npu = w.to(DEVICE)
    b_npu = b.to(DEVICE)
    eta_npu = eta.to(DEVICE)
    h0_npu = h0.to(DEVICE)
    hb0_npu = hb0.to(DEVICE)
    npu_h, npu_v_new, npu_x, npu_y, npu_rstd = chunk_ttt_linear_bwd_h(k_npu, v_npu, w_npu, b_npu, eta_npu, eps_val, h0_npu, hb0_npu, None, bt)
    torch.npu.synchronize()
    expected_outputs = load_fixture(os.path.join(load_dir, 'outputs.pt'), map_location=torch.device('cpu'))
    gpu_h = expected_outputs['output_0']
    gpu_v_new = expected_outputs['output_1']
    gpu_x = expected_outputs['output_2']
    gpu_y = expected_outputs['output_3']
    gpu_rstd = expected_outputs['output_4']
    print('=' * 70)
    print('NPU vs GPU 误差比 (均以 CPU FP64 为参考)')
    print('=' * 70)
    for name, npu_val, gpu_val, cpu_val in [('h', npu_h, gpu_h, cpu_h), ('v_new', npu_v_new, gpu_v_new, cpu_v_new), ('x (kh_hat)', npu_x, gpu_x, cpu_x), ('y (v_new_raw)', npu_y, gpu_y, cpu_y), ('rstd', npu_rstd, gpu_rstd, cpu_rstd)]:
        print(f'\n=== {name} precision ===')
        relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(npu_val.to('cpu').double(), cpu_val)
        relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(gpu_val.to('cpu').double(), cpu_val)
        max_ratio = relative_max_npu / relative_max_gpu
        mean_ratio = relative_mean_npu / relative_mean_gpu
        rmse_ratio = rmse_npu / rmse_gpu
        print(f'  最大相对误差比: {max_ratio:.4f}, 平均相对误差比: {mean_ratio:.4f}, 均方根误差比: {rmse_ratio:.4f}')
        assert max_ratio <= 2, f'{name} max relative error ratio {max_ratio:.4f} exceeds threshold 2'
        assert mean_ratio <= 1.2, f'{name} mean relative error ratio {mean_ratio:.4f} exceeds threshold 1.2'
        assert rmse_ratio <= 1.2, f'{name} rmse ratio {rmse_ratio:.4f} exceeds threshold 1.2'
