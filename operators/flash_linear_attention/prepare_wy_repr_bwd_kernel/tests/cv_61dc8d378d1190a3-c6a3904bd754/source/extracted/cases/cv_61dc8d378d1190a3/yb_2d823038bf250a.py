
from opbench.fla_fixtures import load_fixture
import os
import pytest
import torch
import torch_npu
NPU_ID = os.environ.get('BENCH_NPU_ID', '0')
torch.npu.set_device(int(NPU_ID))
from yb_5deca3efb6cf0b import prepare_wy_repr_bwd_kernel

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
def test_prepare_wy_repr_bwd_kernel_function():
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/prepare_wy_repr_bwd_kernel/'
    try:
        data = load_fixture(os.path.join(load_dir, 'test_prepare_wy_repr_bwd_kernel_v2.pt'), map_location=torch.device('cpu'), weights_only=False)
        datafp64 = load_fixture(os.path.join(load_dir, 'test_prepare_wy_repr_bwd_kernel_fp64_v2.pt'), map_location=torch.device('cpu'), weights_only=False)
    except Exception as e:
        pytest.fail(f'load file failed: {str(e)}')
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    prepare_wy_repr_bwd_kernel[data['grid']](k=input_data['k'], v=input_data['v'], p=input_data['p'], beta=input_data['beta'], g0=input_data['g0'], g=input_data['g'], A=input_data['A'], dw=input_data['dw'], du=input_data['du'], dk=input_data['dk'], dv=input_data['dv'], dp=input_data['dp'], dbeta=input_data['dbeta'], dg0=input_data['dg0'], dg=input_data['dg'], cu_seqlens=input_data['cu_seqlens'], chunk_indices=input_data['chunk_indices'], T=input_data['T'], H=input_data['H'], K=input_data['K'], V=input_data['V'], BT=input_data['BT'], BK=input_data['BK'] // 2, BV=input_data['BV'])
    output_keys = ['dk', 'dv', 'dp', 'dbeta', 'dg0', 'dg']
    for key in output_keys:
        gpu_output_fp64 = datafp64['gpu_output'][key]
        npu_output = input_data[key].cpu()
        gpu_output_fp32 = data['gpu_output'][key]
        print(f'\n=== {key} precision ===')
        relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(npu_output, gpu_output_fp64)
        relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(gpu_output_fp32, gpu_output_fp64)
        max_ratio = relative_max_npu / relative_max_gpu
        mean_ratio = relative_mean_npu / relative_mean_gpu
        rmse_ratio = rmse_npu / rmse_gpu
        print(f'{key} - max relative error ratio: {max_ratio}, mean relative error ratio: {mean_ratio}, rmse ratio: {rmse_ratio}')
        assert max_ratio <= 2, f'{key} max relative error ratio {max_ratio} exceeds threshold 2'
        assert mean_ratio <= 1.2, f'{key} mean relative error ratio {mean_ratio} exceeds threshold 1.2'
        assert rmse_ratio <= 1.2, f'{key} rmse ratio {rmse_ratio} exceeds threshold 1.2'
