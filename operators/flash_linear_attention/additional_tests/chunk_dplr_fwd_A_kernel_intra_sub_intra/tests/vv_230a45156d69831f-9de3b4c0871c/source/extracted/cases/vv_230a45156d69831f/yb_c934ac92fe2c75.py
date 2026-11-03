
from opbench.fla_fixtures import load_fixture
import os
import pytest
import torch
import torch_npu
from yb_d5fb644b2cf2e6 import chunk_dplr_fwd_intra
from yb_22cded5b91bba3 import cal_precision
from yb_22cded5b91bba3 import convert_tensor_with_device_type

def chunk_dplr_fwd_intra_pytorch(q, k, a, b, gi, ge, scale, chunk_size):
    """PyTorch CPU参考实现"""
    B, T, H, K = q.shape
    BT = chunk_size
    NT = (T + BT - 1) // BT
    Aqk = torch.zeros(B, T, H, BT, dtype=torch.float16, device='cpu')
    Aqb = torch.zeros(B, T, H, BT, dtype=torch.float16, device='cpu')
    Aab = torch.zeros(B, T, H, BT, dtype=torch.float32, device='cpu')
    Aak = torch.zeros(B, T, H, BT, dtype=torch.float32, device='cpu')
    qg = torch.zeros_like(q, dtype=torch.float16)
    kg = torch.zeros_like(k, dtype=torch.float16)
    ag = torch.zeros_like(a, dtype=torch.float16)
    bg = torch.zeros_like(b, dtype=torch.float16)
    for b_idx in range(B):
        for h_idx in range(H):
            for t_idx in range(NT):
                t_start = t_idx * BT
                t_end = min((t_idx + 1) * BT, T)
                actual_BT = t_end - t_start
                q_chunk = q[b_idx, t_start:t_end, h_idx, :] * scale
                k_chunk = k[b_idx, t_start:t_end, h_idx, :]
                a_chunk = a[b_idx, t_start:t_end, h_idx, :]
                b_chunk = b[b_idx, t_start:t_end, h_idx, :]
                gi_chunk = gi[b_idx, t_start:t_end, h_idx, :]
                ge_chunk = ge[b_idx, t_start:t_end, h_idx, :]
                last_idx = t_end - 1
                g_last = gi[b_idx, last_idx, h_idx, :]
                g_exp = torch.exp2(gi_chunk)
                g_exp_inv = torch.exp2(-gi_chunk + g_last.unsqueeze(0))
                qg_chunk = q_chunk * g_exp
                kg_chunk = k_chunk * g_exp_inv
                bg_chunk = b_chunk * g_exp_inv
                ag_chunk = a_chunk * torch.exp2(ge_chunk)
                qg[b_idx, t_start:t_end, h_idx, :] = qg_chunk.half()
                kg[b_idx, t_start:t_end, h_idx, :] = kg_chunk.half()
                ag[b_idx, t_start:t_end, h_idx, :] = ag_chunk.half()
                bg[b_idx, t_start:t_end, h_idx, :] = bg_chunk.half()
                for j in range(actual_BT):
                    k_j = k_chunk[j:j + 1, :]
                    gk_j = gi_chunk[j:j + 1, :]
                    b_j = b_chunk[j:j + 1, :]
                    tmp = torch.exp2(gi_chunk - gk_j)
                    A_qk = (q_chunk * k_j * tmp).sum(dim=1)
                    m_i = torch.arange(actual_BT, device='cpu') >= j
                    A_qk = A_qk * m_i.float()
                    A_qb = (q_chunk * b_j * tmp).sum(dim=1)
                    A_qb = A_qb * m_i.float()
                    tmp2 = torch.exp2(ge_chunk - gk_j)
                    A_ak = (a_chunk * k_j * tmp2).sum(dim=1)
                    m_i2 = torch.arange(actual_BT, device='cpu') > j
                    A_ak = A_ak * m_i2.float()
                    A_ab = (a_chunk * b_j * tmp2).sum(dim=1)
                    A_ab = A_ab * m_i2.float()
                    Aqk[b_idx, t_start:t_end, h_idx, j] = A_qk.half()
                    Aqb[b_idx, t_start:t_end, h_idx, j] = A_qb.half()
                    Aak[b_idx, t_start:t_end, h_idx, j] = A_ak
                    Aab[b_idx, t_start:t_end, h_idx, j] = A_ab
    return (Aab, Aqk, Aak, Aqb, qg, kg, ag, bg)

@pytest.mark.functiontest
def test_chunk_dplr_fwd_A_kernel_intra_sub_intra(load_dir):
    data = load_fixture(os.path.join(load_dir, 'test_chunk_dplr_fwd_A_kernel_intra_sub_intra_cuda.pt'), map_location='cpu', weights_only=False)
    input_data = data['input_data']
    q = input_data['q'].float()
    k = input_data['k'].float()
    a = input_data['a'].float()
    b = input_data['b'].float()
    gi = input_data['gi'].float()
    ge = input_data['ge'].float()
    scale = input_data['scale']
    chunk_size = input_data['chunk_size']
    print('=== 精度验证开始 ===\n')
    print(f"输入数据: B={input_data['B']}, T={input_data['T']}, H={input_data['H']}, K={input_data['K']}")
    print(f'chunk_size={chunk_size}, scale={scale}\n')
    print('运行 PyTorch CPU 参考实现...')
    Aab_cpu, Aqk_cpu, Aak_cpu, Aqb_cpu, qg_cpu, kg_cpu, ag_cpu, bg_cpu = chunk_dplr_fwd_intra_pytorch(q, k, a, b, gi, ge, scale, chunk_size)
    print('完成\n')
    print('运行 NPU 实现...')
    device = torch.device('npu')
    input_data_npu = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    Aab_npu, Aqk_npu, Aak_npu, Aqb_npu, qg_npu, kg_npu, ag_npu, bg_npu = chunk_dplr_fwd_intra(q=input_data_npu['q'], k=input_data_npu['k'], a=input_data_npu['a'], b=input_data_npu['b'], gi=input_data_npu['gi'], ge=input_data_npu['ge'], scale=scale, chunk_size=chunk_size, safe_gate=False, cu_seqlens=None, chunk_indices=None)
    print('完成\n')
    gpu_out = data['gpu_output']
    cpu_out = {'Aab': Aab_cpu.float(), 'Aak': Aak_cpu.float(), 'Aqk': Aqk_cpu.float(), 'Aqb': Aqb_cpu.float(), 'qg': qg_cpu.float(), 'kg': kg_cpu.float(), 'ag': ag_cpu.float(), 'bg': bg_cpu.float()}
    gpu_out_tensors = {'Aab': gpu_out['Aab'].float(), 'Aak': gpu_out['Aak'].float(), 'Aqk': gpu_out['Aqk'].float(), 'Aqb': gpu_out['Aqb'].float(), 'qg': gpu_out['qg'].float(), 'kg': gpu_out['kg'].float(), 'ag': gpu_out['ag'].float(), 'bg': gpu_out['bg'].float()}
    npu_out_tensors = {'Aab': Aab_npu.cpu().float(), 'Aak': Aak_npu.cpu().float(), 'Aqk': Aqk_npu.cpu().float(), 'Aqb': Aqb_npu.cpu().float(), 'qg': qg_npu.cpu().float(), 'kg': kg_npu.cpu().float(), 'ag': ag_npu.cpu().float(), 'bg': bg_npu.cpu().float()}
    print('=== 精度指标计算 ===\n')
    results = {}
    for key in cpu_out.keys():
        original_dtype = data['gpu_output'][key].dtype
        print(f'\n{key} (dtype={original_dtype}):')
        print(f'  GPU vs CPU:')
        relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(gpu_out_tensors[key], cpu_out[key])
        print(f'  NPU vs CPU:')
        relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(npu_out_tensors[key], cpu_out[key])
        if original_dtype == torch.float32:
            SMALL_ERR_eps = float(2 ** (-14))
        elif original_dtype == torch.float16:
            SMALL_ERR_eps = float(2 ** (-11))
        elif original_dtype == torch.bfloat16:
            SMALL_ERR_eps = float(2 ** (-8))
        else:
            SMALL_ERR_eps = float(2 ** (-14))
        MARE_ratio = relative_max_npu / (relative_max_gpu + SMALL_ERR_eps)
        MERE_ratio = relative_mean_npu / (relative_mean_gpu + SMALL_ERR_eps)
        RMSE_ratio = rmse_npu / (rmse_gpu + SMALL_ERR_eps)
        results[key] = {'MARE_ratio': MARE_ratio, 'MERE_ratio': MERE_ratio, 'RMSE_ratio': RMSE_ratio, 'dtype': original_dtype}
        print(f'  比值 (NPU/GPU):')
        print(f'    MARE比: {MARE_ratio:.2f}')
        print(f'    MERE比: {MERE_ratio:.2f}')
        print(f'    RMSE比: {RMSE_ratio:.2f}')
    print('=== 总体精度等级评估 ===')
    print('等级标准:')
    print('  L0: MARE比≤10, MERE比≤2.0, RMSE比≤2.0')
    print('  L1: MARE比≤5,  MERE比≤1.5, RMSE比≤1.5')
    print('  L2: MARE比≤2,  MERE比≤1.2, RMSE比≤1.2\n')
    max_MARE_ratio = max((r['MARE_ratio'] for r in results.values()))
    max_MERE_ratio = max((r['MERE_ratio'] for r in results.values()))
    max_RMSE_ratio = max((r['RMSE_ratio'] for r in results.values()))
    print(f'最大MARE比: {max_MARE_ratio:.2f}')
    print(f'最大MERE比: {max_MERE_ratio:.2f}')
    print(f'最大RMSE比: {max_RMSE_ratio:.2f}')
    assert max_MARE_ratio <= 2, 'gw 最大相对误差比 check failed'
    assert max_MERE_ratio <= 1.2, 'gw 平均相对误差比 check failed'
    assert max_RMSE_ratio <= 1.2, 'gw 均方根误差比 check failed'
    print('\n✅ 精度验证通过！')
