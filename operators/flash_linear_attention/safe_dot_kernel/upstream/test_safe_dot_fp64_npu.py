import triton
import triton.language as tl
import torch
import torch_npu
from typing import Optional

def cal_precision(input: torch.tensor, golden: torch.tensor):
    relative_max = torch.max(torch.abs(input-golden)/(torch.abs(golden)+float(1e-7)))
    print(f"max relative error: {relative_max}")
    relative_mean = torch.mean(torch.abs(input-golden)/(torch.abs(golden)+float(1e-7)))
    print(f"mean relative error: {relative_mean}")
    rmse = torch.sqrt(torch.mean((input-golden)**2))
    print(f"rmse: {rmse}")
    mask = (torch.abs(golden) < float(2e-14) ) & (torch.abs(input-golden) > float(2e-30))
    error = torch.abs(input-golden)
    total_error = torch.sum(error[mask])
    print(f"small val error: {total_error}")
    return relative_max, relative_mean, rmse

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

@triton.jit
def safe_dot(a, b, allow_tf32: tl.constexpr = None):
    return tl.dot(a, b, allow_tf32=allow_tf32)

@triton.jit
def safe_dot_kernel(a_ptr, b_ptr, c_ptr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rm = tl.arange(0, BM)
    rn = tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a = tl.load(a_ptr + rm[:, None] * BK + rk[None, :])
    b = tl.load(b_ptr + rk[:, None] * BN + rn[None, :])
    tl.store(c_ptr + rm[:, None] * BN + rn[None, :], safe_dot(a, b, allow_tf32=False))

def test_safe_dot():
    ptfile_path = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/safe_dot_fp64/test_safe_dot_v2.pt'
    ptfile_path1 = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/safe_dot_fp64/test_safe_dot_fp64_v2.pt'
    data = torch.load(ptfile_path, map_location=torch.device('cpu'), weights_only=False)
    datafp64 = torch.load(ptfile_path1, map_location=torch.device('cpu'), weights_only=False)
    
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    
    a = input_data['a']
    b = input_data['b']
    BM = input_data['BM']
    BN = input_data['BN']
    BK = input_data['BK']
    
    c = torch.empty(BM, BN, dtype=a.dtype, device=a.device)
    
    safe_dot_kernel[(1,)](a, b, c, BM, BN, BK)
    
    npu_out = {
        'c': c.cpu(),
    }
    
    # Compare with GPU fp64 output
    gpu_output_c_ptr_fp64 = datafp64['gpu_output']['c']
    npu_output_c_ptr = npu_out['c']
    gpu_output_c_ptr = data['gpu_output']['c']
    
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(npu_output_c_ptr, gpu_output_c_ptr_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(gpu_output_c_ptr, gpu_output_c_ptr_fp64)
    print(f"最大相对误差比 :{relative_max_npu / relative_max_gpu}, 平均相对误差比 :{relative_mean_npu / relative_mean_gpu}, 均方根误差比 :{rmse_npu / rmse_gpu}")
    
    assert relative_max_npu / relative_max_gpu <= 2, "最大相对误差比 check failed"
    assert relative_mean_npu / relative_mean_gpu <= 1.2, "平均相对误差比 check failed"
    assert rmse_npu / rmse_gpu <= 1.2, "均方根误差比 check failed"
    print("Output comparison passed!")


if __name__ == '__main__':
    test_safe_dot()
