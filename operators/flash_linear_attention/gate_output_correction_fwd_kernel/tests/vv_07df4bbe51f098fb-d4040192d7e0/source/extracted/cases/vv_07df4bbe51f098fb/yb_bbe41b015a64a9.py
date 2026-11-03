
from opbench.fla_fixtures import load_fixture
import os
import pytest
import torch
import triton
import torch_npu
from yb_7f11d67c9b6fb9 import gate_output_correction_fwd_kernel
from yb_22cded5b91bba3 import convert_tensor_with_device_type
from yb_22cded5b91bba3 import cal_precision

def gate_output_correction_fwd_triton(o, r, k, r_k, v, g):
    assert r_k.dim() == 2 and r_k.shape[0] == r.shape[-2] and (r_k.shape[1] == r.shape[-1])
    batch_size, seq_len, _ = o.shape
    num_heads, head_dim = (r.shape[-2], r.shape[-1])
    output = torch.empty_like(o)
    for t in range(0, seq_len, 65536):
        T_OFFSET = t
        T_SIZE = min(65536, seq_len - t)

        def grid(meta):
            return (batch_size, triton.cdiv(T_SIZE, meta['BT']), num_heads)
        gate_output_correction_fwd_kernel[grid](o, r, k, r_k, v, g, output, o.stride(0), o.stride(1), r.stride(0), r.stride(1), r.stride(2), v.stride(0), v.stride(1), v.stride(2), r_k.stride(0), T_SIZE, T_OFFSET, num_heads, head_dim, BLOCK_SIZE_D=triton.next_power_of_2(head_dim))
    return output

@pytest.mark.functiontest
def test_gate_output_correction_fwd_kernel(load_dir):
    data = load_fixture(os.path.join(load_dir, 'test_gate_output_correction_fwd_kernel_v2.pt'), map_location=torch.device('cpu'), weights_only=False)
    datafp64 = load_fixture(os.path.join(load_dir, 'test_gate_output_correction_fwd_kernel_fp64_v2.pt'), map_location=torch.device('cpu'), weights_only=False)
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    r = input_data['r']
    k = input_data['k']
    r_k = input_data['r_k']
    v = input_data['v']
    o = input_data['o']
    g = input_data['g']
    output = gate_output_correction_fwd_triton(o, r, k, r_k, v, g)
    npu_out = {'output': output.cpu()}
    for key in datafp64['gpu_output']:
        if key in npu_out:
            gpu_output = data['gpu_output'][key]
            gpu_output_fp64 = datafp64['gpu_output'][key]
            npu_output = npu_out[key]
            print(f'Comparing {key}:')
            relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(npu_output, gpu_output_fp64)
            relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(gpu_output, gpu_output_fp64)
            print(f'最大相对误差比 :{relative_max_npu / relative_max_gpu}, 平均相对误差比 :{relative_mean_npu / relative_mean_gpu}, 均方根误差比 :{rmse_npu / rmse_gpu}')
            assert relative_max_npu / relative_max_gpu <= 2, '最大相对误差比 check failed'
            assert relative_mean_npu / relative_mean_gpu <= 1.2, '平均相对误差比 check failed'
            assert rmse_npu / rmse_gpu <= 1.2, '均方根误差比 check failed'
    print('Output comparison passed!')
