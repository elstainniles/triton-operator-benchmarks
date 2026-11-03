
from opbench.fla_fixtures import load_fixture
import sys
import os
import torch
from yb_5b9c84961deac4 import chunk_gsa_fwd_k
from yb_edcd590a6e2c1d import chunk_gsa_fwd_k_torch

def convert_tensor_with_device_type(indata: dict, device_type: str):
    target_device = torch.device(device_type)
    outdata = {}
    for key, value in indata.items():
        if isinstance(value, torch.Tensor):
            if value.device.type != target_device.type:
                outdata[key] = value.to(target_device)
            else:
                outdata[key] = value
        else:
            outdata[key] = value
    return outdata

def test_chunk_gsa_fwd_third_cmp():
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    ptfile_path = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_gsa_fwd_k/test_chunk_gsa_fwd_k.pt'
    data = load_fixture(ptfile_path, map_location='cpu', weights_only=False)
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    A, h, ht, o = chunk_gsa_fwd_k(q=input_data['q'], k=input_data['k'], v=input_data['v'], g=input_data['g'], h0=input_data['h0'], output_final_state=input_data['output_final_state'], scale=input_data['scale'], cu_seqlens=None, chunk_size=input_data['BT'])
    print(f'\n运行Torch Native实现...')
    A_torch, h_torch, _, o_torch = chunk_gsa_fwd_k_torch(q=input_data['q'], k=input_data['k'], v=input_data['v'], g=input_data['g'], scale=input_data['scale'], BT=input_data['BT'])
    gpu_fp32_data = load_fixture('/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_gsa_fwd_k/test_chunk_gsa_fwd_k_v2.pt', map_location='cpu', weights_only=False)
    gpu_output = gpu_fp32_data['gpu_output']
    A_gpu = gpu_output['A']
    h_gpu = gpu_output['h']
    o_gpu = gpu_output['o']
    device = 'npu'
    mare_npu, mere_npu, rmse_npu = cal_precision(A.to(torch.float32), A_torch.to(torch.float32).to(device))
    mare_gpu, mere_gpu, rmse_gpu = cal_precision(A_gpu.to(device).to(torch.float32), A_torch.to(torch.float32).to(device))
    mare_thd, mere_thd, rmse_thd = (10, 2, 2)
    mare_thd, mere_thd, rmse_thd = (5, 1.5, 1.5)
    mare_thd, mere_thd, rmse_thd = (2, 1.2, 1.2)
    print('\n============================ A =========================')
    print(mare_npu / mare_gpu, mere_npu / mere_gpu, rmse_npu / rmse_gpu)
    assert mare_npu / mare_gpu <= mare_thd
    assert mere_npu / mere_gpu <= mere_thd
    assert rmse_npu / rmse_gpu <= rmse_thd
    mare_npu, mere_npu, rmse_npu = cal_precision(h.to(torch.float32), h_torch.to(torch.float32).to(device))
    mare_gpu, mere_gpu, rmse_gpu = cal_precision(h_gpu.to(device).to(torch.float32), h_torch.to(torch.float32).to(device))
    print('\n============================ h =========================')
    print(mare_npu / mare_gpu, mere_npu / mere_gpu, rmse_npu / rmse_gpu)
    assert mare_npu / mare_gpu <= mare_thd
    assert mere_npu / mere_gpu <= mere_thd
    assert rmse_npu / rmse_gpu <= rmse_thd
    mare_npu, mere_npu, rmse_npu = cal_precision(o.to(torch.float32), o_torch.to(torch.float32).to(device))
    mare_gpu, mere_gpu, rmse_gpu = cal_precision(o_gpu.to(device).to(torch.float32), o_torch.to(torch.float32).to(device))
    print('\n============================ o =========================')
    print(mare_npu / mare_gpu, mere_npu / mere_gpu, rmse_npu / rmse_gpu)
    assert mare_npu / mare_gpu <= mare_thd
    assert mere_npu / mere_gpu <= mere_thd
    assert rmse_npu / rmse_gpu <= rmse_thd

def cal_precision(input: torch.tensor, golden: torch.tensor):
    relative_max = torch.max(torch.abs(input - golden) / (torch.abs(golden) + float(1e-07)))
    relative_mean = torch.mean(torch.abs(input - golden) / (torch.abs(golden) + float(1e-07)))
    rmse = torch.sqrt(torch.mean((input - golden) ** 2))
    compare_tensor(input, golden)
    return (relative_max, relative_mean, rmse)

def compare_tensor(npu_tensor, golden_tensor):
    abs_diff = torch.abs(npu_tensor - golden_tensor)
    abs_golden = torch.abs(golden_tensor)
    rel_diff = abs_diff / (abs_golden + 1e-10)
    max_abs_error = abs_diff.max().item()
    max_rel_error = rel_diff.max().item()
    mean_abs_error = abs_diff.mean().item()
    mean_rel_error = rel_diff.mean().item()
    rmse = torch.sqrt((abs_diff ** 2).mean()).item()
    ret = torch.allclose(npu_tensor, golden_tensor, 0.005, 0.005)
    print(f'\n=====<<< allclose: {ret}:')
    print(f'  Max absolute error: {max_abs_error:.6e}')
    print(f'  Max relative error: {max_rel_error:.6e}')
    print(f'  Mean absolute error: {mean_abs_error:.6e}')
    print(f'  Mean relative error: {mean_rel_error:.6e}')
    print(f'  RMSE: {rmse:.6e}')
