# coding=utf-8
from opbench.fla_fixtures import load_fixture
import os
import pytest
import torch
import torch_npu
from yb_3e77d2739161a0 import grpo_bwd_kernel
NPU_ID = os.environ.get('BENCH_NPU_ID', '0')
torch.npu.set_device(int(NPU_ID))
kernel_name = 'grpo_bwd_kernel'

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
def test_grpo_bwd_kernel_functiontest(load_dir):
    data = load_fixture(os.path.join(load_dir, f'{kernel_name}_in.pt'), map_location=torch.device('cpu'), weights_only=False)
    datafp64 = load_fixture(os.path.join(load_dir, f'{kernel_name}_fp64_out.pt'), map_location=torch.device('cpu'), weights_only=False)
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    grpo_bwd_kernel[data['grid']](**input_data)
    gpu_output_c_ptr_fp64 = datafp64['gpu_output']['dlogits_ptr']
    npu_output_c_ptr = input_data['dlogits_ptr'].cpu()
    gpu_output_c_ptr = data['gpu_output']['dlogits_ptr']
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(npu_output_c_ptr, gpu_output_c_ptr_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(gpu_output_c_ptr, gpu_output_c_ptr_fp64)
    max_ratio = relative_max_npu / relative_max_gpu
    mean_ratio = relative_mean_npu / relative_mean_gpu
    rmse_ratio = rmse_npu / rmse_gpu
    print(f'最大相对误差比 :{max_ratio}, 平均相对误差比 :{mean_ratio}, 均方根误差比 :{rmse_ratio}')
    assert max_ratio <= 2, f'最大相对误差比 {max_ratio} 超过阈值 2'
    assert mean_ratio <= 1.2, f'平均相对误差比 {mean_ratio} 超过阈值 1.2'
    assert rmse_ratio <= 1.2, f'均方根相对误差比 {rmse_ratio} 超过阈值 1.2'
