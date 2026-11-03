
from opbench.fla_fixtures import load_fixture
import sys
import os
import torch
from yb_ab1792945eaa4c import chunk_gsa_bwd_k

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

def test_chunk_gsa_bwd_k_npu():
    ptfile_path = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_gsa_bwd_k/test_chunk_gsa_bwd_k.pt'
    data = load_fixture(ptfile_path, map_location='cpu', weights_only=False)
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    dq, dk, dv, dg, dh0, _, _ = chunk_gsa_bwd_k(q=input_data['q'], k=input_data['k'], v=input_data['v'], g=input_data['g'], h=input_data['h'], h0=input_data['h0'], o=input_data['o'], do=input_data['do'], dht=input_data['dht'], dg=input_data['dg'], scale=input_data['scale'], cu_seqlens=None, chunk_size=input_data['BT'])
    npu_output = {'dq': dq.cpu().clone(), 'dk': dk.cpu().clone(), 'dv': dv.cpu().clone(), 'dg': dg.cpu().clone(), 'dh0': torch.empty(0) if dh0 is None else dh0.cpu().clone()}
    gpu_fp32_data = load_fixture('/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_gsa_bwd_k/test_chunk_gsa_bwd_k_v2.pt', map_location='cpu', weights_only=False)
    gpu_fp32_output = gpu_fp32_data['gpu_output']
    print('\n' + '=' * 60)
    print('NPU vs GPU fp32 Precision Comparison')
    print('=' * 60)
    for key in ['dq', 'dk', 'dv', 'dg']:
        npu_tensor = npu_output[key].float()
        gpu_tensor = gpu_fp32_output[key].float()
        abs_diff = torch.abs(npu_tensor - gpu_tensor)
        abs_gpu = torch.abs(gpu_tensor)
        rel_diff = abs_diff / (abs_gpu + 1e-10)
        max_abs_error = abs_diff.max().item()
        max_rel_error = rel_diff.max().item()
        mean_abs_error = abs_diff.mean().item()
        mean_rel_error = rel_diff.mean().item()
        rmse = torch.sqrt((abs_diff ** 2).mean()).item()
        assert torch.allclose(npu_tensor, gpu_tensor, 0.005, 0.005)
        print(f'\n{key}:')
        print(f'  Max absolute error: {max_abs_error:.6e}')
        print(f'  Max relative error: {max_rel_error:.6e}')
        print(f'  Mean absolute error: {mean_abs_error:.6e}')
        print(f'  Mean relative error: {mean_rel_error:.6e}')
        print(f'  RMSE: {rmse:.6e}')
