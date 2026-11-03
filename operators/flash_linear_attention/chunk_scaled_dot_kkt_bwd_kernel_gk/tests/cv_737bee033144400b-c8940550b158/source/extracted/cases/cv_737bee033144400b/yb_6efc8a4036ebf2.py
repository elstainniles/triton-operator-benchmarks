# coding=utf-8
from opbench.fla_fixtures import load_fixture
import os
import pytest
import torch
import torch_npu
NPU_ID = os.environ.get('BENCH_NPU_ID', '0')
torch.npu.set_device(int(NPU_ID))
from yb_3a11c7d24ffaf7 import chunk_scaled_dot_kkt_bwd_kernel_gk

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

def cal_precision(input: torch.tensor, golden: torch.tensor):
    relative_max = torch.max(torch.abs(input - golden) / (torch.abs(golden) + float(1e-07)))
    print(f'max relative error: {relative_max}')
    relative_mean = torch.mean(torch.abs(input - golden) / (torch.abs(golden) + float(1e-07)))
    print(f'mean relative error: {relative_mean}')
    rmse = torch.sqrt(torch.mean((input - golden) ** 2))
    print(f'rmse: {rmse}')
    mask = (torch.abs(golden) < float(2e-14)) & (torch.abs(input - golden) > float(2e-30))
    error = torch.abs(input - golden)
    total_error = torch.sum(error[mask])
    print(f'small val error: {total_error}')
    return (relative_max, relative_mean, rmse)

@pytest.mark.functiontest
def test_chunk_scaled_dot_kkt_bwd_kernel_gk_function():
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_scaled_dot_kkt_bwd_kernel_gk/'
    try:
        data = load_fixture(os.path.join(load_dir, 'test_chunk_scaled_dot_kkt_bwd_kernel_gk_v2.pt'), map_location=torch.device('cpu'), weights_only=False)
        datafp64 = load_fixture(os.path.join(load_dir, 'test_chunk_scaled_dot_kkt_bwd_kernel_gk_fp64_v2.pt'), map_location=torch.device('cpu'), weights_only=False)
    except Exception as e:
        pytest.fail(f'load file failed: {str(e)}')
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    chunk_scaled_dot_kkt_bwd_kernel_gk[data['grid']](**input_data)
    gpu_output_dk_fp64 = datafp64['gpu_output']['dk']
    gpu_output_dg_fp64 = datafp64['gpu_output']['dg']
    gpu_output_db_fp64 = datafp64['gpu_output']['db']
    npu_output_dk = input_data['dk'].cpu()
    npu_output_dg = input_data['dg'].cpu()
    npu_output_db = input_data['db'].cpu()
    gpu_output_dk = data['gpu_output']['dk']
    gpu_output_dg = data['gpu_output']['dg']
    gpu_output_db = data['gpu_output']['db']
    print('=== dk precision ===')
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(npu_output_dk, gpu_output_dk_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(gpu_output_dk, gpu_output_dk_fp64)
    max_ratio = relative_max_npu / relative_max_gpu
    mean_ratio = relative_mean_npu / relative_mean_gpu
    rmse_ratio = rmse_npu / rmse_gpu
    print(f'max relative error ratio: {max_ratio}, mean relative error ratio: {mean_ratio}, rmse ratio: {rmse_ratio}')
    assert max_ratio <= 2, f'dk max relative error ratio {max_ratio} exceeds threshold 2'
    assert mean_ratio <= 1.2, f'dk mean relative error ratio {mean_ratio} exceeds threshold 1.2'
    assert rmse_ratio <= 1.2, f'dk rmse ratio {rmse_ratio} exceeds threshold 1.2'
    print('\n=== dg precision ===')
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(npu_output_dg, gpu_output_dg_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(gpu_output_dg, gpu_output_dg_fp64)
    max_ratio = relative_max_npu / relative_max_gpu
    mean_ratio = relative_mean_npu / relative_mean_gpu
    rmse_ratio = rmse_npu / rmse_gpu
    print(f'max relative error ratio: {max_ratio}, mean relative error ratio: {mean_ratio}, rmse ratio: {rmse_ratio}')
    assert max_ratio <= 2, f'dg max relative error ratio {max_ratio} exceeds threshold 2'
    assert mean_ratio <= 1.2, f'dg mean relative error ratio {mean_ratio} exceeds threshold 1.2'
    assert rmse_ratio <= 1.2, f'dg rmse ratio {rmse_ratio} exceeds threshold 1.2'
    print('\n=== db precision ===')
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(npu_output_db, gpu_output_db_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(gpu_output_db, gpu_output_db_fp64)
    max_ratio = relative_max_npu / relative_max_gpu
    mean_ratio = relative_mean_npu / relative_mean_gpu
    rmse_ratio = rmse_npu / rmse_gpu
    print(f'max relative error ratio: {max_ratio}, mean relative error ratio: {mean_ratio}, rmse ratio: {rmse_ratio}')
    assert max_ratio <= 2, f'db max relative error ratio {max_ratio} exceeds threshold 2'
    assert mean_ratio <= 1.2, f'db mean relative error ratio {mean_ratio} exceeds threshold 1.2'
    assert rmse_ratio <= 1.2, f'db rmse ratio {rmse_ratio} exceeds threshold 1.2'
