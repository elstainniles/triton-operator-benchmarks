# coding=utf-8
from opbench.fla_fixtures import load_fixture
import os
import pytest
import torch
import torch_npu
from yb_712432704460d6 import chunk_global_cumsum_vector
NPU_ID = os.environ.get('BENCH_NPU_ID', '0')
torch.npu.set_device(int(NPU_ID))
DEVICE = f'npu:{torch.npu.current_device()}'
kernel_name = 'chunk_global_cumsum_vector_kernel'

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
def test_chunk_global_cumsum_vector_kernel_functiontest(load_dir):
    inputs = load_fixture(os.path.join(load_dir, f'{kernel_name}_in.pt'), map_location=torch.device('cpu'), weights_only=False)
    gpu_outputs = load_fixture(os.path.join(load_dir, f'{kernel_name}_out.pt'), map_location=torch.device('cpu'), weights_only=False)
    s_cpu = inputs['s']
    cpu_output = s_cpu.to(torch.float64).cumsum(dim=1)
    s = s_cpu.to(DEVICE)
    npu_output = chunk_global_cumsum_vector(s)
    if isinstance(npu_output, tuple):
        for i, out in enumerate(npu_output):
            npu_out = out.cpu()
            gpu_out = gpu_outputs[f'output_{i}']
            cpu_out = cpu_output[i] if isinstance(cpu_output, (tuple, list)) else cpu_output
            relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(npu_out, cpu_out)
            relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(gpu_out, cpu_out)
            max_ratio = relative_max_npu / relative_max_gpu
            mean_ratio = relative_mean_npu / relative_mean_gpu
            rmse_ratio = rmse_npu / rmse_gpu
            print(f'Output {i} - 最大相对误差比 :{max_ratio}, 平均相对误差比 :{mean_ratio}, 均方根误差比 :{rmse_ratio}')
            assert max_ratio <= 2, f'Output {i} 最大相对误差比 {max_ratio} 超过阈值 2'
            assert mean_ratio <= 1.2, f'Output {i} 平均相对误差比 {mean_ratio} 超过阈值 1.2'
            assert rmse_ratio <= 1.2, f'Output {i} 均方根相对误差比 {rmse_ratio} 超过阈值 1.2'
    else:
        npu_out = npu_output.cpu()
        gpu_out = gpu_outputs['output']
        cpu_out = cpu_output
        relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(npu_out, cpu_out)
        relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(gpu_out, cpu_out)
        max_ratio = relative_max_npu / relative_max_gpu
        mean_ratio = relative_mean_npu / relative_mean_gpu
        rmse_ratio = rmse_npu / rmse_gpu
        print(f'最大相对误差比 :{max_ratio}, 平均相对误差比 :{mean_ratio}, 均方根误差比 :{rmse_ratio}')
        assert max_ratio <= 2, f'最大相对误差比 {max_ratio} 超过阈值 2'
        assert mean_ratio <= 1.2, f'平均相对误差比 {mean_ratio} 超过阈值 1.2'
        assert rmse_ratio <= 1.2, f'均方根相对误差比 {rmse_ratio} 超过阈值 1.2'
    print(f'[{kernel_name}] Three-party precision check (NPU vs GPU, ref=CPU) completed successfully')
