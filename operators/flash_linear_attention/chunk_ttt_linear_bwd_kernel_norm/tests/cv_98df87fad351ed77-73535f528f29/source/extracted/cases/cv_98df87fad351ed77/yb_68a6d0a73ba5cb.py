
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

def naive_chunk_ttt_linear_bwd_norm(q, k, v, v_new, x, y, rstd, w, b, eta, h0, hb0, h, dht, dhbt, dv_new, do, scale, chunk_size, dtype=torch.float64):
    B, T, H, K = q.shape
    V = do.shape[-1]
    BT = chunk_size
    NT = math.ceil(T / BT)
    dh = torch.zeros(B, NT, H, K, V, dtype=dtype)
    dhb = torch.zeros(B, NT, H, 1, V, dtype=dtype)
    dv = torch.zeros(B, T, H, V, dtype=dtype)
    dk = torch.zeros(B, T, H, K, dtype=dtype)
    dw_per_bh = torch.zeros(B, H, V, dtype=dtype)
    db_per_bh = torch.zeros(B, H, V, dtype=dtype)
    dh0 = torch.zeros(B, H, K, V, dtype=dtype) if h0 is not None else None
    dhb0 = torch.zeros(B, H, 1, V, dtype=dtype) if hb0 is not None else None
    for i_b in range(B):
        for i_h in range(H):
            b_dh = torch.zeros(K, V, dtype=dtype)
            b_dhb = torch.zeros(V, dtype=dtype)
            if dht is not None:
                b_dh = b_dh + dht[i_b, i_h].to(dtype)
            if dhbt is not None:
                b_dhb = b_dhb + dhbt[i_b, i_h].reshape(V).to(dtype)
            _w = w[i_h].to(dtype)
            _b = b[i_h].to(dtype)
            b_dw = torch.zeros(V, dtype=dtype)
            b_db = torch.zeros(V, dtype=dtype)
            for i_t in range(NT - 1, -1, -1):
                t_start = i_t * BT
                t_end = min((i_t + 1) * BT, T)
                dh[i_b, i_t, i_h] = b_dh
                dhb[i_b, i_t, i_h, 0, :] = b_dhb
                _k = k[i_b, t_start:t_end, i_h].to(dtype)
                _dv_new = dv_new[i_b, t_start:t_end, i_h].to(dtype).clone()
                eta_last = eta[i_b, t_end - 1, i_h, 0].to(dtype)
                _dv_new = _dv_new - eta_last * (_k @ b_dh) - eta_last * b_dhb.unsqueeze(0)
                _v_new = v_new[i_b, t_start:t_end, i_h].to(dtype)
                _x = x[i_b, t_start:t_end, i_h].to(dtype)
                _y = y[i_b, t_start:t_end, i_h].to(dtype)
                _rstd = rstd[i_b, t_start:t_end, i_h, 0:1].to(dtype)
                _dy = _rstd * (_dv_new * V - _dv_new.sum(dim=-1, keepdim=True) - _x * (_dv_new * _x).sum(dim=-1, keepdim=True)) / V
                _dx = -_rstd * (_dv_new * (_x * _y).sum(dim=-1, keepdim=True) + _y * (_dv_new * _x).sum(dim=-1, keepdim=True)) / V
                _drstd = (_dv_new * _v_new / _rstd).sum(dim=-1, keepdim=True)
                _v = v[i_b, t_start:t_end, i_h].to(dtype)
                _dv = -_w.unsqueeze(0) * _dy
                _dk = _w.unsqueeze(0) * _dy
                b_dw = b_dw + (2 * _w.unsqueeze(0) * _x * _dy + (_b.unsqueeze(0) - _v + _k) * _dy).sum(dim=0)
                b_db = b_db + (_w.unsqueeze(0) * _dy).sum(dim=0)
                _dx = _dx + _w.unsqueeze(0) * _w.unsqueeze(0) * _dy
                _q = q[i_b, t_start:t_end, i_h].to(dtype)
                _h = h[i_b, i_t, i_h].to(dtype)
                _do = do[i_b, t_start:t_end, i_h].to(dtype)
                _dkh = _rstd * (V * _dx - _dx.sum(dim=-1, keepdim=True) - _x * (_x * _dx).sum(dim=-1, keepdim=True)) / V
                _dkh = _dkh - _rstd * _rstd * _drstd * _x / V
                _dk = _dk + _dkh @ _h
                b_dh = b_dh + (scale * _q).T @ _do + _k.T @ _dkh
                b_dhb = b_dhb + (_do + _dkh).sum(dim=0)
                dv[i_b, t_start:t_end, i_h] = _dv
                dk[i_b, t_start:t_end, i_h] = _dk
            dw_per_bh[i_b, i_h] = b_dw
            db_per_bh[i_b, i_h] = b_db
            if h0 is not None:
                dh0[i_b, i_h] = b_dh
            if hb0 is not None:
                dhb0[i_b, i_h, 0, :] = b_dhb
    dw_final = dw_per_bh.sum(dim=0)
    db_final = db_per_bh.sum(dim=0)
    return (dh, dhb, dh0, dhb0, dv, dk, dw_final, db_final)

@pytest.mark.functiontest
def test_chunk_ttt_linear_bwd_kernel_norm():
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_ttt_linear_bwd_kernel_norm'
    q = load_fixture(os.path.join(load_dir, 'q.pt'), map_location=torch.device('cpu'))
    k = load_fixture(os.path.join(load_dir, 'k.pt'), map_location=torch.device('cpu'))
    v = load_fixture(os.path.join(load_dir, 'v.pt'), map_location=torch.device('cpu'))
    v_new = load_fixture(os.path.join(load_dir, 'v_new.pt'), map_location=torch.device('cpu'))
    x = load_fixture(os.path.join(load_dir, 'x.pt'), map_location=torch.device('cpu'))
    y = load_fixture(os.path.join(load_dir, 'y.pt'), map_location=torch.device('cpu'))
    rstd = load_fixture(os.path.join(load_dir, 'rstd.pt'), map_location=torch.device('cpu'))
    w = load_fixture(os.path.join(load_dir, 'w.pt'), map_location=torch.device('cpu'))
    b = load_fixture(os.path.join(load_dir, 'b.pt'), map_location=torch.device('cpu'))
    eta = load_fixture(os.path.join(load_dir, 'eta.pt'), map_location=torch.device('cpu'))
    h0 = load_fixture(os.path.join(load_dir, 'h0.pt'), map_location=torch.device('cpu'))
    hb0 = load_fixture(os.path.join(load_dir, 'hb0.pt'), map_location=torch.device('cpu'))
    h = load_fixture(os.path.join(load_dir, 'h.pt'), map_location=torch.device('cpu'))
    dht = load_fixture(os.path.join(load_dir, 'dht.pt'), map_location=torch.device('cpu'))
    dhbt = load_fixture(os.path.join(load_dir, 'dhbt.pt'), map_location=torch.device('cpu'))
    dv_new = load_fixture(os.path.join(load_dir, 'dv_new.pt'), map_location=torch.device('cpu'))
    do_tensor = load_fixture(os.path.join(load_dir, 'do_tensor.pt'), map_location=torch.device('cpu'))
    B, T, H, D = q.shape
    K, V = (D, D)
    scale = D ** (-0.5)
    chunk_size = 16
    print(f'Input shapes: q={q.shape}, k={k.shape}, v={v.shape}, h={h.shape}')
    print(f'Config: B={B}, T={T}, H={H}, D={D}, chunk_size={chunk_size}, scale={scale}')
    cpu_dh, cpu_dhb, cpu_dh0, cpu_dhb0, cpu_dv, cpu_dk, cpu_dw, cpu_db = naive_chunk_ttt_linear_bwd_norm(q.detach().clone().to(torch.float64), k.detach().clone().to(torch.float64), v.detach().clone().to(torch.float64), v_new.detach().clone().to(torch.float64), x.detach().clone().to(torch.float64), y.detach().clone().to(torch.float64), rstd.detach().clone().to(torch.float64), w.detach().clone().to(torch.float64), b.detach().clone().to(torch.float64), eta.detach().clone().to(torch.float64), h0.detach().clone().to(torch.float64), hb0.detach().clone().to(torch.float64), h.detach().clone().to(torch.float64), dht.detach().clone().to(torch.float64), dhbt.detach().clone().to(torch.float64), dv_new.detach().clone().to(torch.float64), do_tensor.detach().clone().to(torch.float64), scale, chunk_size, dtype=torch.float64)
    from yb_7bff0dd87eff27 import chunk_ttt_linear_bwd_norm
    q_npu = q.to(DEVICE)
    k_npu = k.to(DEVICE)
    v_npu = v.to(DEVICE)
    v_new_npu = v_new.to(DEVICE)
    x_npu = x.to(DEVICE)
    y_npu = y.to(DEVICE)
    rstd_npu = rstd.to(DEVICE)
    w_npu = w.to(DEVICE)
    b_npu = b.to(DEVICE)
    eta_npu = eta.to(DEVICE)
    h0_npu = h0.to(DEVICE)
    hb0_npu = hb0.to(DEVICE)
    h_npu = h.to(DEVICE)
    dht_npu = dht.to(DEVICE)
    dhbt_npu = dhbt.to(DEVICE)
    dv_new_npu = dv_new.to(DEVICE)
    do_npu = do_tensor.to(DEVICE)
    npu_dh, npu_dhb, npu_dh0, npu_dhb0, npu_dv, npu_dk, npu_dw, npu_db = chunk_ttt_linear_bwd_norm(q_npu, k_npu, v_npu, v_new_npu, x_npu, y_npu, rstd_npu, w_npu, b_npu, eta_npu, h0_npu, hb0_npu, h_npu, dht_npu, dhbt_npu, dv_new_npu, do_npu, scale, None, chunk_size)
    torch.npu.synchronize()
    gpu_dh = load_fixture(os.path.join(load_dir, 'output_0.pt'), map_location=torch.device('cpu'))
    gpu_dhb = load_fixture(os.path.join(load_dir, 'output_1.pt'), map_location=torch.device('cpu'))
    gpu_dh0 = load_fixture(os.path.join(load_dir, 'output_2.pt'), map_location=torch.device('cpu'))
    gpu_dhb0 = load_fixture(os.path.join(load_dir, 'output_3.pt'), map_location=torch.device('cpu'))
    gpu_dv = load_fixture(os.path.join(load_dir, 'output_4.pt'), map_location=torch.device('cpu'))
    gpu_dk = load_fixture(os.path.join(load_dir, 'output_5.pt'), map_location=torch.device('cpu'))
    gpu_dw = load_fixture(os.path.join(load_dir, 'output_6.pt'), map_location=torch.device('cpu'))
    gpu_db = load_fixture(os.path.join(load_dir, 'output_7.pt'), map_location=torch.device('cpu'))
    print('=' * 70)
    print('三方对比: NPU vs GPU (均以 CPU FP64 为参考)')
    print('=' * 70)
    third_part_cmp(npu_dh, gpu_dh, cpu_dh, 'dh')
    third_part_cmp(npu_dhb, gpu_dhb, cpu_dhb, 'dhb')
    third_part_cmp(npu_dh0, gpu_dh0, cpu_dh0, 'dh0')
    third_part_cmp(npu_dhb0, gpu_dhb0, cpu_dhb0, 'dhb0')
    third_part_cmp(npu_dv, gpu_dv, cpu_dv, 'dv')
    third_part_cmp(npu_dk, gpu_dk, cpu_dk, 'dk')
    third_part_cmp(npu_dw, gpu_dw, cpu_dw, 'dw')
    third_part_cmp(npu_db, gpu_db, cpu_db, 'db')
