
from opbench.fla_fixtures import load_fixture
import os
import pytest
import torch
import torch_npu
import triton
from yb_0c04dfbe64c0ce import elementwise_mul_kernel
from yb_22cded5b91bba3 import compare_data_precision
from yb_22cded5b91bba3 import convert_tensor_with_device_type

@pytest.mark.functiontest
def test_elementwise_mul_kernel(load_path):
    data = load_fixture(os.path.join(load_path, 'test_elementwise_mul_kernel_cuda.pt'), map_location=torch.device('cpu'), weights_only=False)
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    x = input_data['x'].clone()
    g = input_data['g']
    N = input_data['N']
    D = input_data['D']
    B = min(65536 // 2, triton.next_power_of_2(D))
    elementwise_mul_kernel[triton.cdiv(N * D, B),](x=x, g=g, N=N * D, B=B)
    npu_out = {}
    npu_out['x'] = x
    compare_data_precision(data['gpu_output'], npu_out, device_type='cpu')
    print('Test elementwise_mul_kernel passed!')
