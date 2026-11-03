
from opbench.fla_fixtures import load_fixture
import os
import pytest
import torch
import torch_npu
from yb_83cc11308a4a36 import fused_recurrent_rwkv4_backward
from yb_22cded5b91bba3 import cal_precision
from yb_22cded5b91bba3 import validate_cmp

def fused_recurrent_rwkv4_backward_pytorch_fp64(w, u, k, v, state, grad_wkv, grad_state):
    """PyTorch CPU float64参考实现"""
    bsz, tsz, chans = k.shape
    gw = torch.zeros(bsz, chans, dtype=torch.float64, device='cpu')
    gu = torch.zeros(bsz, chans, dtype=torch.float64, device='cpu')
    gk = torch.zeros(bsz, tsz, chans, dtype=torch.float64, device='cpu')
    gv = torch.zeros(bsz, tsz, chans, dtype=torch.float64, device='cpu')
    gstate = torch.zeros(bsz, 3, 1, chans, dtype=torch.float64, device='cpu')
    for b in range(bsz):
        galpha = grad_state[b, 0, 0, :].double()
        gbeta = grad_state[b, 1, 0, :].double()
        geps = grad_state[b, 2, 0, :].double()
        for t in range(tsz - 1, -1, -1):
            kt = k[b, t, :].double()
            vt = v[b, t, :].double()
            alpha_prev = state[b, 0, t, :].double()
            beta_prev = state[b, 1, t, :].double()
            eps_prev = state[b, 2, t, :].double()
            alpha_curr = state[b, 0, t + 1, :].double()
            beta_curr = state[b, 1, t + 1, :].double()
            eps_curr = state[b, 2, t + 1, :].double()
            ukt = u + kt
            tau = torch.maximum(ukt, eps_prev)
            e1 = torch.exp(eps_prev - tau)
            e2 = torch.exp(ukt - tau)
            euke = torch.exp(ukt + eps_prev - 2 * tau)
            denom = e1 * beta_prev + e2
            denom_sq = denom * denom
            gwkvt = grad_wkv[b, t, :].double()
            guk = gwkvt * e2 * (e1 * beta_prev * vt - e1 * alpha_prev) / denom_sq
            gu[b, :] += guk
            gk[b, t, :] = guk
            gv[b, t, :] = gwkvt * e2 / denom
            galpha_wkv = gwkvt * e1 / denom
            gbeta_wkv = -gwkvt * e1 * (e2 * vt + e1 * alpha_prev) / denom_sq
            geps_wkv_denom = e1 * beta_prev + e2
            geps_wkv = gwkvt * euke * (alpha_prev - vt * beta_prev) / (geps_wkv_denom * geps_wkv_denom)
            e1 = torch.exp(w + eps_prev - eps_curr)
            e2 = torch.exp(kt - eps_curr)
            galpha_we = galpha * e1 * alpha_prev
            gw[b, :] += galpha_we
            gk[b, t, :] += galpha * e2 * vt
            gv[b, t, :] += galpha * e2
            geps += galpha * -alpha_curr
            gbeta_we = gbeta * e1 * beta_prev
            gw[b, :] += gbeta_we
            gk[b, t, :] += gbeta * e2
            geps += gbeta * -beta_curr
            geps_mask = w + eps_prev > kt
            geps_we = torch.where(geps_mask, geps, torch.zeros_like(geps))
            gw[b, :] += geps_we
            gk[b, t, :] += torch.where(geps_mask, torch.zeros_like(geps), geps)
            galpha = galpha * e1 + galpha_wkv
            gbeta = gbeta * e1 + gbeta_wkv
            geps = galpha_we + gbeta_we + geps_we + geps_wkv
        gstate[b, 0, 0, :] = galpha
        gstate[b, 1, 0, :] = gbeta
        gstate[b, 2, 0, :] = geps
    gw = gw * w
    return (gw.sum(0), gu.sum(0), gk, gv, gstate)

@pytest.mark.functiontest
def test_fused_recurrent_rwkv4_backward(load_dir):
    inputs_dict = load_fixture(os.path.join(load_dir, 'inputs_rwkv4_backward.pt'), map_location='cpu', weights_only=False)
    device = torch.device('npu')
    w_processed = inputs_dict['w_processed'].to(device)
    u = inputs_dict['u'].to(device)
    k = inputs_dict['k'].to(device)
    v = inputs_dict['v'].to(device)
    state = inputs_dict['state'].to(device)
    grad_wkv = inputs_dict['grad_wkv'].to(device)
    grad_state = inputs_dict['grad_state'].to(device)
    gw, gu, gk, gv, gstate = fused_recurrent_rwkv4_backward(w_processed, u, k, v, state, grad_wkv, grad_state)
    outputs_dict = load_fixture(os.path.join(load_dir, 'outputs_rwkv4_backward.pt'), map_location='cpu', weights_only=False)
    gw_gpu = outputs_dict['gw'].to(device)
    gu_gpu = outputs_dict['gu'].to(device)
    gk_gpu = outputs_dict['gk'].to(device)
    gv_gpu = outputs_dict['gv'].to(device)
    gstate_gpu = outputs_dict['gstate'].to(device)
    print('Computing PyTorch CPU float64 reference...')
    w_processed_fp64 = inputs_dict['w_processed'].to(torch.float64)
    u_fp64 = inputs_dict['u'].to(torch.float64)
    k_fp64 = inputs_dict['k'].to(torch.float64)
    v_fp64 = inputs_dict['v'].to(torch.float64)
    state_fp64 = inputs_dict['state'].to(torch.float64)
    grad_wkv_fp64 = inputs_dict['grad_wkv'].to(torch.float64)
    grad_state_fp64 = inputs_dict['grad_state'].to(torch.float64)
    gw_fp64, gu_fp64, gk_fp64, gv_fp64, gstate_fp64 = fused_recurrent_rwkv4_backward_pytorch_fp64(w_processed_fp64, u_fp64, k_fp64, v_fp64, state_fp64, grad_wkv_fp64, grad_state_fp64)
    print('PyTorch CPU float64 reference computed.\n')
    gw_gpu_fp64 = gw_fp64.to(device)
    gu_gpu_fp64 = gu_fp64.to(device)
    gk_gpu_fp64 = gk_fp64.to(device)
    gv_gpu_fp64 = gv_fp64.to(device)
    gstate_gpu_fp64 = gstate_fp64.to(device)
    print('\nComparing gw with expected values...')
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(gw, gw_gpu_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(gw_gpu, gw_gpu_fp64)
    print(f'最大相对误差比 :{relative_max_npu / relative_max_gpu}, 平均相对误差比 :{relative_mean_npu / relative_mean_gpu}, 均方根误差比 :{rmse_npu / rmse_gpu}')
    assert relative_max_npu / relative_max_gpu <= 2, 'gw 最大相对误差比 check failed'
    assert relative_mean_npu / relative_mean_gpu <= 1.2, 'gw 平均相对误差比 check failed'
    assert rmse_npu / rmse_gpu <= 1.2, 'gw 均方根误差比 check failed'
    print('\nComparing gu with expected values...')
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(gu, gu_gpu_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(gu_gpu, gu_gpu_fp64)
    if relative_max_gpu == 0 and relative_max_npu == 0:
        print('GPU和NPU误差均为0，精度完美匹配')
        print(f'最大相对误差比 :N/A (均为0), 平均相对误差比 :N/A (均为0), 均方根误差比 :N/A (均为0)')
    elif relative_max_gpu == 0:
        print(f'NPU最大相对误差: {relative_max_npu:.6e}')
        print(f'GPU误差为0，检查NPU绝对误差是否在精度范围内')
        validate_cmp(dtype=str(gu.dtype).split('.')[-1], y_ref=gu, y_cal=gu_gpu, device_type='cpu')
    else:
        print(f'最大相对误差比 :{relative_max_npu / relative_max_gpu}, 平均相对误差比 :{relative_mean_npu / relative_mean_gpu}, 均方根误差比 :{rmse_npu / rmse_gpu}')
        assert relative_max_npu / relative_max_gpu <= 2, 'gu 最大相对误差比 check failed'
        assert relative_mean_npu / relative_mean_gpu <= 1.2, 'gu 平均相对误差比 check failed'
        assert rmse_npu / rmse_gpu <= 1.2, 'gu 均方根误差比 check failed'
    print('\nComparing gk with expected values...')
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(gk, gk_gpu_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(gk_gpu, gk_gpu_fp64)
    print(f'最大相对误差比 :{relative_max_npu / relative_max_gpu}, 平均相对误差比 :{relative_mean_npu / relative_mean_gpu}, 均方根误差比 :{rmse_npu / rmse_gpu}')
    assert relative_max_npu / relative_max_gpu <= 2, 'gk 最大相对误差比 check failed'
    assert relative_mean_npu / relative_mean_gpu <= 1.2, 'gk 平均相对误差比 check failed'
    assert rmse_npu / rmse_gpu <= 1.2, 'gk 均方根误差比 check failed'
    print('\nComparing gv with expected values...')
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(gv, gv_gpu_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(gv_gpu, gv_gpu_fp64)
    print(f'最大相对误差比 :{relative_max_npu / relative_max_gpu}, 平均相对误差比 :{relative_mean_npu / relative_mean_gpu}, 均方根误差比 :{rmse_npu / rmse_gpu}')
    assert relative_max_npu / relative_max_gpu <= 2, 'gv 最大相对误差比 check failed'
    assert relative_mean_npu / relative_mean_gpu <= 1.2, 'gv 平均相对误差比 check failed'
    assert rmse_npu / rmse_gpu <= 1.2, 'gv 均方根误差比 check failed'
    print('\nComparing gstate with expected values...')
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(gstate, gstate_gpu_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(gstate_gpu, gstate_gpu_fp64)
    print(f'最大相对误差比 :{relative_max_npu / relative_max_gpu}, 平均相对误差比 :{relative_mean_npu / relative_mean_gpu}, 均方根误差比 :{rmse_npu / rmse_gpu}')
    assert relative_max_npu / relative_max_gpu <= 2, 'gstate 最大相对误差比 check failed'
    assert relative_mean_npu / relative_mean_gpu <= 1.2, 'gstate 平均相对误差比 check failed'
    assert rmse_npu / rmse_gpu <= 1.2, 'gstate 均方根误差比 check failed'
    print('Test fused_recurrent_rwkv4_backward passed!')
