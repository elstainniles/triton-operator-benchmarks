
from opbench.fla_fixtures import load_fixture
import os
os.environ['FLA_USE_FAST_OPS'] = '0'
import torch
import torch_npu
from yb_b0825468b432fa import chunk_generalized_iplr_delta_rule_fwd_h
from yb_b0825468b432fa import prepare_wy_repr_fwd
from yb_f95f816dd5c2a2 import compare_data_precision
from yb_f95f816dd5c2a2 import convert_tensor_with_device_type

def chunk_generalized_iplr_delta_rule_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, a: torch.Tensor, b: torch.Tensor, scale: float, initial_state: torch.Tensor, output_final_state: bool, cu_seqlens: torch.LongTensor | None=None, chunk_size: int=64, chunk_indices: torch.LongTensor | None=None):
    w, u, _ = prepare_wy_repr_fwd(a=a, b=b, k=k, v=v, cu_seqlens=cu_seqlens, chunk_size=chunk_size, chunk_indices=chunk_indices)
    h, v_new, final_state = chunk_generalized_iplr_delta_rule_fwd_h(k=k, v=v, b=b, w=w, u=u, initial_state=initial_state, output_final_state=output_final_state, cu_seqlens=cu_seqlens, chunk_size=chunk_size, chunk_indices=chunk_indices)
    return (h, v_new, final_state)

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

def fp64_precision_compare(npu_out, gpu_out, fp64_data):
    for key in fp64_data.keys():
        if key in npu_out:
            gpu_output = gpu_out[key]
            gpu_output_fp64 = fp64_data[key]
            npu_output = npu_out[key]
            print(f'Comparing {key}:')
            relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(npu_output, gpu_output_fp64)
            relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(gpu_output, gpu_output_fp64)
            MARE_ratio = relative_max_npu / relative_max_gpu if relative_max_gpu > 1e-15 else 1.0
            MERE_ratio = relative_mean_npu / relative_mean_gpu if relative_mean_gpu > 1e-15 else 1.0
            RMSE_ratio = rmse_npu / rmse_gpu if rmse_gpu > 1e-15 else 1.0
            print(f'最大相对误差比 :{relative_max_npu / relative_max_gpu}, 平均相对误差比 :{relative_mean_npu / relative_mean_gpu}, 均方根误差比 :{rmse_npu / rmse_gpu}')
            assert MARE_ratio, '最大相对误差比 check failed'
            assert MERE_ratio <= 1.2, '平均相对误差比 check failed'
            assert RMSE_ratio <= 1.2, '均方根误差比 check failed'

def test_chunk_generalized_iplr_delta_rule_fwd_kernel_h():
    ptfile_path = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_generalized_iplr_delta_rule_fwd_kernel_h/test_chunk_generalized_iplr_delta_rule_fwd_kernel_h_cuda.pt'
    data = load_fixture(ptfile_path, map_location=torch.device('cpu'), weights_only=False)
    ptfile_path1 = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_generalized_iplr_delta_rule_fwd_kernel_h/test_chunk_generalized_iplr_delta_rule_fwd_kernel_h_cuda_fp64.pt'
    datafp64 = load_fixture(ptfile_path1, map_location=torch.device('cpu'), weights_only=False)
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    h, v_new, final_state = chunk_generalized_iplr_delta_rule_fwd(q=input_data['q'], k=input_data['k'], v=input_data['v'], a=input_data['a'], b=input_data['b'], scale=input_data['scale'], initial_state=input_data['initial_state'], output_final_state=input_data['output_final_state'], cu_seqlens=input_data['cu_seqlens'], chunk_size=input_data['chunk_size'], chunk_indices=input_data['chunk_indices'])
    npu_out = {}
    npu_out['h'] = h.cpu()
    npu_out['v_new'] = v_new.cpu()
    npu_out['final_state'] = final_state.cpu()
    fp64_precision_compare(npu_out, data['gpu_output'], datafp64)
    print('Test chunk_generalized_iplr_delta_rule_fwd_kernel_h passed!')
