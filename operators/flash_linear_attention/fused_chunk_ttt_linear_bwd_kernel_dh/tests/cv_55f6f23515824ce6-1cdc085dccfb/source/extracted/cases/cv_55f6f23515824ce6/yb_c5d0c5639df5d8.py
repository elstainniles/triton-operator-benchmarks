
from opbench.fla_fixtures import load_fixture
import os
import pytest
import torch
import torch_npu
NPU_ID = os.environ.get('BENCH_NPU_ID', '0')
torch.npu.set_device(int(NPU_ID))
from yb_14d6a471f4f932 import fused_chunk_ttt_linear_bwd_kernel_dh

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
def test_fused_chunk_ttt_linear_bwd_kernel_dh_function():
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/fused_chunk_ttt_linear_bwd_kernel_dh/'
    try:
        data = load_fixture(os.path.join(load_dir, 'test_fused_chunk_ttt_linear_bwd_kernel_dh_v2.pt'), map_location=torch.device('cpu'), weights_only=False)
        datafp64 = load_fixture(os.path.join(load_dir, 'test_fused_chunk_ttt_linear_bwd_kernel_dh_fp64_v2.pt'), map_location=torch.device('cpu'), weights_only=False)
    except Exception as e:
        pytest.fail(f'load file failed: {str(e)}')
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    fused_chunk_ttt_linear_bwd_kernel_dh[data['grid']](**input_data)
    gpu_output_dk_fp64 = datafp64['gpu_output']['dk']
    npu_output_dk = input_data['dk'].cpu()
    gpu_output_dk = data['gpu_output']['dk']
    print('=== dk precision ===')
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(npu_output_dk, gpu_output_dk_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(gpu_output_dk, gpu_output_dk_fp64)
    max_ratio = relative_max_npu / relative_max_gpu
    mean_ratio = relative_mean_npu / relative_mean_gpu
    rmse_ratio = rmse_npu / rmse_gpu
    print(f'最大相对误差比: {max_ratio}, 平均相对误差比: {mean_ratio}, 均方根误差比: {rmse_ratio}')
    assert max_ratio <= 2, f'dk max relative error ratio {max_ratio} exceeds threshold 2'
    assert mean_ratio <= 1.2, f'dk mean relative error ratio {mean_ratio} exceeds threshold 1.2'
    assert rmse_ratio <= 1.2, f'dk rmse ratio {rmse_ratio} exceeds threshold 1.2'
    gpu_output_dv_fp64 = datafp64['gpu_output']['dv']
    npu_output_dv = input_data['dv'].cpu()
    gpu_output_dv = data['gpu_output']['dv']
    print('\n=== dv precision ===')
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(npu_output_dv, gpu_output_dv_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(gpu_output_dv, gpu_output_dv_fp64)
    max_ratio = relative_max_npu / relative_max_gpu
    mean_ratio = relative_mean_npu / relative_mean_gpu
    rmse_ratio = rmse_npu / rmse_gpu
    print(f'最大相对误差比: {max_ratio}, 平均相对误差比: {mean_ratio}, 均方根误差比: {rmse_ratio}')
    assert max_ratio <= 2, f'dv max relative error ratio {max_ratio} exceeds threshold 2'
    assert mean_ratio <= 1.2, f'dv mean relative error ratio {mean_ratio} exceeds threshold 1.2'
    assert rmse_ratio <= 1.2, f'dv rmse ratio {rmse_ratio} exceeds threshold 1.2'
    gpu_output_de_fp64 = datafp64['gpu_output']['de']
    npu_output_de = input_data['de'].cpu()
    gpu_output_de = data['gpu_output']['de']
    print('\n=== de precision ===')
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(npu_output_de, gpu_output_de_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(gpu_output_de, gpu_output_de_fp64)
    max_ratio = relative_max_npu / relative_max_gpu
    mean_ratio = relative_mean_npu / relative_mean_gpu
    rmse_ratio = rmse_npu / rmse_gpu
    print(f'最大相对误差比: {max_ratio}, 平均相对误差比: {mean_ratio}, 均方根误差比: {rmse_ratio}')
    assert max_ratio <= 2, f'de max relative error ratio {max_ratio} exceeds threshold 2'
    assert mean_ratio <= 1.2, f'de mean relative error ratio {mean_ratio} exceeds threshold 1.2'
    assert rmse_ratio <= 1.2, f'de rmse ratio {rmse_ratio} exceeds threshold 1.2'
    gpu_output_dw_fp64 = datafp64['gpu_output']['dw'].sum(dim=0)
    npu_output_dw = input_data['dw'].cpu().sum(dim=0)
    gpu_output_dw = data['gpu_output']['dw'].sum(dim=0)
    print('\n=== dw precision ===')
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(npu_output_dw, gpu_output_dw_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(gpu_output_dw, gpu_output_dw_fp64)
    max_ratio = relative_max_npu / relative_max_gpu
    mean_ratio = relative_mean_npu / relative_mean_gpu
    rmse_ratio = rmse_npu / rmse_gpu
    print(f'最大相对误差比: {max_ratio}, 平均相对误差比: {mean_ratio}, 均方根误差比: {rmse_ratio}')
    assert max_ratio <= 2, f'dw max relative error ratio {max_ratio} exceeds threshold 2'
    assert mean_ratio <= 1.2, f'dw mean relative error ratio {mean_ratio} exceeds threshold 1.2'
    assert rmse_ratio <= 1.2, f'dw rmse ratio {rmse_ratio} exceeds threshold 1.2'
    gpu_output_db_fp64 = datafp64['gpu_output']['db'].sum(dim=0)
    npu_output_db = input_data['db'].cpu().sum(dim=0)
    gpu_output_db = data['gpu_output']['db'].sum(dim=0)
    print('\n=== db precision ===')
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(npu_output_db, gpu_output_db_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(gpu_output_db, gpu_output_db_fp64)
    max_ratio = relative_max_npu / relative_max_gpu
    mean_ratio = relative_mean_npu / relative_mean_gpu
    rmse_ratio = rmse_npu / rmse_gpu
    print(f'最大相对误差比: {max_ratio}, 平均相对误差比: {mean_ratio}, 均方根误差比: {rmse_ratio}')
    assert max_ratio <= 2, f'db max relative error ratio {max_ratio} exceeds threshold 2'
    assert mean_ratio <= 1.2, f'db mean relative error ratio {mean_ratio} exceeds threshold 1.2'
    assert rmse_ratio <= 1.2, f'db rmse ratio {rmse_ratio} exceeds threshold 1.2'
