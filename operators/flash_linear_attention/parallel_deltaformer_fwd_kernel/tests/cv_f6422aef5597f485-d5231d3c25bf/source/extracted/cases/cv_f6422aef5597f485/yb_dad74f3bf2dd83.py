
from opbench.fla_fixtures import load_fixture
import os
import pytest
import torch
import torch_npu
from yb_a87c34e2ef2580 import parallel_deltaformer_fwd_kernel
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
def test_parallel_deltaformer_fwd_kernel_function():
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/parallel_deltaformer_fwd_kernel'
    try:
        data = load_fixture(os.path.join(load_dir, 'test_parallel_deltaformer_fwd_kernel_v2.pt'), map_location=torch.device('cpu'), weights_only=False)
        datafp64 = load_fixture(os.path.join(load_dir, 'test_parallel_deltaformer_fwd_kernel_fp64_v2.pt'), map_location=torch.device('cpu'), weights_only=False)
    except Exception as e:
        pytest.fail(f'load file failed: {str(e)}')
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    parallel_deltaformer_fwd_kernel[data['grid']](input_data['q'], input_data['k'], input_data['v'], input_data['u'], input_data['w'], input_data['lse'], input_data['beta'], input_data['H'], input_data['T'], input_data['C'], input_data['D'], input_data['qk_scale'])
    gpu_output_w_fp64 = datafp64['gpu_output']['w']
    npu_output_w = input_data['w'].cpu()
    gpu_output_w = data['gpu_output']['w']
    gpu_output_lse_fp64 = datafp64['gpu_output']['lse']
    npu_output_lse = input_data['lse'].cpu()
    gpu_output_lse = data['gpu_output']['lse']
    print('=== w precision ===')
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(npu_output_w, gpu_output_w_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(gpu_output_w, gpu_output_w_fp64)
    max_ratio = relative_max_npu / relative_max_gpu
    mean_ratio = relative_mean_npu / relative_mean_gpu
    rmse_ratio = rmse_npu / rmse_gpu
    print(f'w - max relative error ratio: {max_ratio}, mean relative error ratio: {mean_ratio}, rmse ratio: {rmse_ratio}')
    assert max_ratio <= 2, f'w max relative error ratio {max_ratio} exceeds threshold 2'
    assert mean_ratio <= 1.2, f'w mean relative error ratio {mean_ratio} exceeds threshold 1.2'
    assert rmse_ratio <= 1.2, f'w rmse ratio {rmse_ratio} exceeds threshold 1.2'
    print('\n=== lse precision ===')
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(npu_output_lse, gpu_output_lse_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(gpu_output_lse, gpu_output_lse_fp64)
    max_ratio = relative_max_npu / relative_max_gpu
    mean_ratio = relative_mean_npu / relative_mean_gpu
    rmse_ratio = rmse_npu / rmse_gpu
    print(f'lse - max relative error ratio: {max_ratio}, mean relative error ratio: {mean_ratio}, rmse ratio: {rmse_ratio}')
    assert max_ratio <= 2, f'lse max relative error ratio {max_ratio} exceeds threshold 2'
    assert mean_ratio <= 1.2, f'lse mean relative error ratio {mean_ratio} exceeds threshold 1.2'
    assert rmse_ratio <= 1.2, f'lse rmse ratio {rmse_ratio} exceeds threshold 1.2'
