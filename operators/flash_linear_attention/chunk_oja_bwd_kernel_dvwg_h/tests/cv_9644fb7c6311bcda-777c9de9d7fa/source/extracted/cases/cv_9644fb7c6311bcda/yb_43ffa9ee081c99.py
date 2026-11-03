
from opbench.fla_fixtures import load_fixture
import os
import pytest
import torch
import torch_npu
from yb_5a9aa3bd6a5163 import chunk_oja_bwd_kernel_dvwg_h
NPU_ID = os.environ.get('BENCH_NPU_ID', '0')
torch.npu.set_device(int(NPU_ID))

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
def test_chunk_oja_bwd_kernel_dvwg_h_function():
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_oja_bwd_kernel_dvwg_h/'
    try:
        data = load_fixture(os.path.join(load_dir, 'test_chunk_oja_bwd_kernel_dvwg_h_v2.pt'), map_location=torch.device('cpu'), weights_only=False)
        datafp64 = load_fixture(os.path.join(load_dir, 'test_chunk_oja_bwd_kernel_dvwg_h_fp64_v2.pt'), map_location=torch.device('cpu'), weights_only=False)
    except Exception as e:
        pytest.fail(f'load file failed: {str(e)}')
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    chunk_oja_bwd_kernel_dvwg_h[data['grid']](k=input_data['k'], v=input_data['v'], gv=input_data['gv'], h=input_data['h'], dh=input_data['dh'], dk=input_data['dk'], dw=input_data['dw'], dv=input_data['dv'], dgv_last=input_data['dgv_last'], dgk=input_data['dgk'], cu_seqlens=input_data['cu_seqlens'], chunk_indices=input_data['chunk_indices'], T=input_data['T'], H=input_data['H'], K=input_data['K'], V=input_data['V'], BT=input_data['BT'], BK=input_data['BK'], BV=input_data['BV'])
    gpu_output_dv_fp64 = datafp64['gpu_output']['dv']
    npu_output_dv = input_data['dv'].cpu()
    gpu_output_dv = data['gpu_output']['dv']
    gpu_output_dw_fp64 = datafp64['gpu_output']['dw']
    npu_output_dw = input_data['dw'].cpu()
    gpu_output_dw = data['gpu_output']['dw']
    gpu_output_dgv_last_fp64 = datafp64['gpu_output']['dgv_last']
    npu_output_dgv_last = input_data['dgv_last'].cpu()
    gpu_output_dgv_last = data['gpu_output']['dgv_last']
    print('=== dv precision ===')
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(npu_output_dv, gpu_output_dv_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(gpu_output_dv, gpu_output_dv_fp64)
    max_ratio = relative_max_npu / relative_max_gpu
    mean_ratio = relative_mean_npu / relative_mean_gpu
    rmse_ratio = rmse_npu / rmse_gpu
    print(f'dv - max relative error ratio: {max_ratio}, mean relative error ratio: {mean_ratio}, rmse ratio: {rmse_ratio}')
    assert max_ratio <= 2, f'dv max relative error ratio {max_ratio} exceeds threshold 2'
    assert mean_ratio <= 1.2, f'dv mean relative error ratio {mean_ratio} exceeds threshold 1.2'
    assert rmse_ratio <= 1.2, f'dv rmse ratio {rmse_ratio} exceeds threshold 1.2'
    print('\n=== dw precision ===')
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(npu_output_dw, gpu_output_dw_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(gpu_output_dw, gpu_output_dw_fp64)
    max_ratio = relative_max_npu / relative_max_gpu
    mean_ratio = relative_mean_npu / relative_mean_gpu
    rmse_ratio = rmse_npu / rmse_gpu
    print(f'dw - max relative error ratio: {max_ratio}, mean relative error ratio: {mean_ratio}, rmse ratio: {rmse_ratio}')
    assert max_ratio <= 2, f'dw max relative error ratio {max_ratio} exceeds threshold 2'
    assert mean_ratio <= 1.2, f'dw mean relative error ratio {mean_ratio} exceeds threshold 1.2'
    assert rmse_ratio <= 1.2, f'dw rmse ratio {rmse_ratio} exceeds threshold 1.2'
    print('\n=== dgv_last precision ===')
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(npu_output_dgv_last, gpu_output_dgv_last_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(gpu_output_dgv_last, gpu_output_dgv_last_fp64)
    max_ratio = relative_max_npu / relative_max_gpu
    mean_ratio = relative_mean_npu / relative_mean_gpu
    rmse_ratio = rmse_npu / rmse_gpu
    print(f'dgv_last - max relative error ratio: {max_ratio}, mean relative error ratio: {mean_ratio}, rmse ratio: {rmse_ratio}')
    assert max_ratio <= 2, f'dgv_last max relative error ratio {max_ratio} exceeds threshold 2'
    assert mean_ratio <= 1.2, f'dgv_last mean relative error ratio {mean_ratio} exceeds threshold 1.2'
    assert rmse_ratio <= 1.2, f'dgv_last rmse ratio {rmse_ratio} exceeds threshold 1.2'
