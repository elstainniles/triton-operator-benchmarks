
from opbench.fla_fixtures import load_fixture
import os
import pytest
import torch
import torch_npu
from yb_2854025b508f9d import Rwkv7FusedAddcmul
from yb_22cded5b91bba3 import compare_data_precision
from yb_22cded5b91bba3 import convert_tensor_with_device_type

@pytest.mark.functiontest
def test_fused_addcmul_fwd_kernel(load_dir):
    inputs_dict = load_fixture(os.path.join(load_dir, 'input_torch.pt'), map_location=torch.device('cpu'))
    outputs_dict = load_fixture(os.path.join(load_dir, 'output_torch.pt'), map_location=torch.device('cpu'))
    input_data = convert_tensor_with_device_type(inputs_dict, device_type='npu')
    hidden_states = input_data['hidden_states']
    delta = input_data['delta']
    x_r = input_data['x_r']
    x_w = input_data['x_w']
    x_k = input_data['x_k']
    x_v = input_data['x_v']
    x_a = input_data['x_a']
    x_g = input_data['x_g']
    oxr, oxw, oxk, oxv, oxa, oxg = Rwkv7FusedAddcmul.apply(hidden_states, delta, x_r, x_w, x_k, x_v, x_a, x_g)
    npu_out = {}
    npu_out['oxr'] = oxr
    npu_out['oxw'] = oxw
    npu_out['oxk'] = oxk
    npu_out['oxv'] = oxv
    npu_out['oxa'] = oxa
    npu_out['oxg'] = oxg
    compare_data_precision(outputs_dict, npu_out, device_type='cpu')
    print('Test fused_addcmul_fwd_kernel passed!')
