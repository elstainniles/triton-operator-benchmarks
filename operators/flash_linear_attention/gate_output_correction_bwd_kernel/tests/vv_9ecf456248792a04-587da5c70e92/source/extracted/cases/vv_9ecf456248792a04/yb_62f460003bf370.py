
from opbench.fla_fixtures import load_fixture
import os
import pytest
import torch
import torch_npu
from yb_22cded5b91bba3 import convert_tensor_with_device_type
from yb_22cded5b91bba3 import cal_precision
from yb_7f11d67c9b6fb9 import gate_output_correction_backward_triton

@pytest.mark.functiontest
def test_gate_output_correction_bwd_kernel(load_dir):
    data = load_fixture(os.path.join(load_dir, 'test_gate_output_correction_bwd_kernel_v2.pt'), map_location=torch.device('cpu'), weights_only=False)
    datafp64 = load_fixture(os.path.join(load_dir, 'test_gate_output_correction_bwd_kernel_fp64_v2.pt'), map_location=torch.device('cpu'), weights_only=False)
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    r = input_data['r']
    k = input_data['k']
    r_k = input_data['r_k']
    v = input_data['v']
    o = input_data['o']
    grad_output = input_data['grad_output']
    g = input_data['g']
    grad_o, grad_r, grad_k, grad_r_k, grad_v, grad_g = gate_output_correction_backward_triton(grad_output, o, r, k, r_k, v, g)
    npu_out = {'grad_o': grad_o.cpu(), 'grad_r': grad_r.cpu(), 'grad_k': grad_k.cpu(), 'grad_r_k': grad_r_k.cpu(), 'grad_v': grad_v.cpu(), 'grad_g': grad_g.cpu()}
    for key in datafp64['gpu_output']:
        if key in npu_out:
            gpu_output = data['gpu_output'][key]
            gpu_output_fp64 = datafp64['gpu_output'][key]
            npu_output = npu_out[key]
            print(f'Comparing {key}:')
            relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(npu_output, gpu_output_fp64)
            relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(gpu_output, gpu_output_fp64)
            print(f'最大相对误差比 :{relative_max_npu / relative_max_gpu}, 平均相对误差比 :{relative_mean_npu / relative_mean_gpu}, 均方根误差比 :{rmse_npu / rmse_gpu}')
            assert relative_max_npu / relative_max_gpu <= 5, '最大相对误差比 check failed'
            assert relative_mean_npu / relative_mean_gpu <= 1.5, '平均相对误差比 check failed'
            assert rmse_npu / rmse_gpu <= 1.5, '均方根误差比 check failed'
    print('Output comparison passed!')
