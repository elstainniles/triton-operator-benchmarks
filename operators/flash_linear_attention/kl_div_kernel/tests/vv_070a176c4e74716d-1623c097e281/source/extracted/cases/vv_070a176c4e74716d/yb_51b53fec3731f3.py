# coding=utf-8
from opbench.fla_fixtures import load_fixture
import os
import pytest
import torch
import torch_npu
from yb_7d3ff6d12bf094 import kl_div_kernel
NPU_ID = os.environ.get('BENCH_NPU_ID', '0')
torch.npu.set_device(int(NPU_ID))
kernel_name = 'kl_div_kernel'

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
def test_kl_div_kernel_functiontest(load_dir):
    data = load_fixture(os.path.join(load_dir, f'{kernel_name}_in.pt'), map_location=torch.device('cpu'), weights_only=False)
    datafp64 = load_fixture(os.path.join(load_dir, f'{kernel_name}_fp64_out.pt'), map_location=torch.device('cpu'), weights_only=False)
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    kl_div_kernel[data['grid']](logits=input_data['logits'], target_logits=input_data['target_logits'], loss=input_data['loss'], s_logits=input_data['s_logits'], s_loss=input_data['s_loss'], reduction=input_data['reduction'], N=input_data['N'], V=input_data['V'], BV=input_data['BV'])
    gpu_output_logits_fp64 = datafp64['gpu_output']['logits']
    gpu_output_loss_fp64 = datafp64['gpu_output']['loss']
    npu_output_logits = input_data['logits'].cpu()
    npu_output_loss = input_data['loss'].cpu()
    gpu_output_logits = data['gpu_output']['logits']
    gpu_output_loss = data['gpu_output']['loss']
    print('=== Logits precision ===')
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(npu_output_logits, gpu_output_logits_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(gpu_output_logits, gpu_output_logits_fp64)
    max_ratio = relative_max_npu / relative_max_gpu
    mean_ratio = relative_mean_npu / relative_mean_gpu
    rmse_ratio = rmse_npu / rmse_gpu
    print(f'Max relative error ratio: {max_ratio}, Mean relative error ratio: {mean_ratio}, RMSE ratio: {rmse_ratio}')
    assert max_ratio <= 2, f'Logits 最大相对误差比 {max_ratio} 超过阈值 2'
    assert mean_ratio <= 1.2, f'Logits 平均相对误差比 {mean_ratio} 超过阈值 1.2'
    assert rmse_ratio <= 1.2, f'Logits 均方根相对误差比 {rmse_ratio} 超过阈值 1.2'
    print('\n=== Loss precision ===')
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(npu_output_loss, gpu_output_loss_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(gpu_output_loss, gpu_output_loss_fp64)
    max_ratio = relative_max_npu / relative_max_gpu
    mean_ratio = relative_mean_npu / relative_mean_gpu
    rmse_ratio = rmse_npu / rmse_gpu
    print(f'Max relative error ratio: {max_ratio}, Mean relative error ratio: {mean_ratio}, RMSE ratio: {rmse_ratio}')
    assert max_ratio <= 2, f'Loss 最大相对误差比 {max_ratio} 超过阈值 2'
    assert mean_ratio <= 1.2, f'Loss 平均相对误差比 {mean_ratio} 超过阈值 1.2'
    assert rmse_ratio <= 1.2, f'Loss 均方根相对误差比 {rmse_ratio} 超过阈值 1.2'
