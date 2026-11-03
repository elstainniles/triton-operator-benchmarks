
from opbench.fla_fixtures import load_fixture
import os
import pytest
import torch
import torch_npu
from yb_2854025b508f9d import addcmul_bwd1
from yb_22cded5b91bba3 import compare_data_precision
from yb_22cded5b91bba3 import convert_tensor_with_device_type

@pytest.mark.functiontest
def test_addcmul_bwd_kernel1(load_path):
    data = load_fixture(os.path.join(load_path, 'test_addcmul_bwd_kernel1_cuda.pt'), map_location=torch.device('cpu'), weights_only=False)
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    use_xg = input_data['use_xg']
    g_hiddn, g_delta = addcmul_bwd1(input_data['d_xr'], input_data['d_xw'], input_data['d_xk'], input_data['d_xv'], input_data['d_xa'], input_data['d_xg'], input_data['x_r'], input_data['x_w'], input_data['x_k'], input_data['x_v'], input_data['x_a'], input_data['x_g'], input_data['hidden_states'], input_data['delta'], use_xg)
    npu_out = {}
    npu_out['g_hiddn'] = g_hiddn
    npu_out['g_delta'] = g_delta
    compare_data_precision(data['gpu_output'], npu_out, device_type='cpu')
    print('Test addcmul_bwd_kernel1 passed!')
