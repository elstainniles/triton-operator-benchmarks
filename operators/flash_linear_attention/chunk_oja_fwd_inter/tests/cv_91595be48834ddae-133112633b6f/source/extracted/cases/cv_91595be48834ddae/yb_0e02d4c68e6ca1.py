
from opbench.fla_fixtures import load_fixture
import os
import pytest
import torch
import torch_npu
import triton
from yb_8b1f9f64c8c620 import chunk_oja_fwd_inter
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

def make_grid(data):
    input_data = data['input_data']
    T = input_data['T']
    B = input_data['q'].shape[0]
    HQ = input_data['q'].shape[2]
    V = input_data['gv'].shape[-1]
    BT = min(64, max(16, triton.next_power_of_2(T)))
    NT = triton.cdiv(T, BT)

    def grid(meta):
        return (triton.cdiv(V, meta['BV']), NT, B * HQ)
    return grid

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
def test_chunk_oja_fwd_inter_kernel_function():
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_oja_fwd_inter/'
    try:
        data = load_fixture(os.path.join(load_dir, 'test_chunk_oja_fwd_inter_v2.pt'), map_location=torch.device('cpu'), weights_only=False)
        datafp64 = load_fixture(os.path.join(load_dir, 'test_chunk_oja_fwd_inter_fp64_v2.pt'), map_location=torch.device('cpu'), weights_only=False)
    except Exception as e:
        pytest.fail(f'load file failed: {str(e)}')
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    grid = make_grid(data)
    chunk_oja_fwd_inter[grid](**input_data)
    gpu_output_o_fp64 = datafp64['gpu_output']['o']
    gpu_output_A_fp64 = datafp64['gpu_output']['A']
    npu_output_o = input_data['o'].cpu()
    npu_output_A = input_data['A'].cpu()
    gpu_output_o = data['gpu_output']['o']
    gpu_output_A = data['gpu_output']['A']
    print("=== Output 'o' precision ===")
    relative_max_npu_o, relative_mean_npu_o, rmse_npu_o = cal_precision(npu_output_o, gpu_output_o_fp64)
    relative_max_gpu_o, relative_mean_gpu_o, rmse_gpu_o = cal_precision(gpu_output_o, gpu_output_o_fp64)
    max_ratio_o = relative_max_npu_o / relative_max_gpu_o
    mean_ratio_o = relative_mean_npu_o / relative_mean_gpu_o
    rmse_ratio_o = rmse_npu_o / rmse_gpu_o
    print(f'最大相对误差比: {max_ratio_o}, 平均相对误差比: {mean_ratio_o}, 均方根误差比: {rmse_ratio_o}')
    assert max_ratio_o <= 2, f'o max relative error ratio {max_ratio_o} exceeds threshold 2'
    assert mean_ratio_o <= 1.2, f'o mean relative error ratio {mean_ratio_o} exceeds threshold 1.2'
    assert rmse_ratio_o <= 1.2, f'o rmse ratio {rmse_ratio_o} exceeds threshold 1.2'
    print("\n=== Output 'A' precision ===")
    relative_max_npu_A, relative_mean_npu_A, rmse_npu_A = cal_precision(npu_output_A, gpu_output_A_fp64)
    relative_max_gpu_A, relative_mean_gpu_A, rmse_gpu_A = cal_precision(gpu_output_A, gpu_output_A_fp64)
    max_ratio_A = relative_max_npu_A / relative_max_gpu_A
    mean_ratio_A = relative_mean_npu_A / relative_mean_gpu_A
    rmse_ratio_A = rmse_npu_A / rmse_gpu_A
    print(f'最大相对误差比: {max_ratio_A}, 平均相对误差比: {mean_ratio_A}, 均方根误差比: {rmse_ratio_A}')
    assert max_ratio_A <= 2, f'A max relative error ratio {max_ratio_A} exceeds threshold 2'
    assert mean_ratio_A <= 1.2, f'A mean relative error ratio {mean_ratio_A} exceeds threshold 1.2'
    assert rmse_ratio_A <= 1.2, f'A rmse ratio {rmse_ratio_A} exceeds threshold 1.2'
