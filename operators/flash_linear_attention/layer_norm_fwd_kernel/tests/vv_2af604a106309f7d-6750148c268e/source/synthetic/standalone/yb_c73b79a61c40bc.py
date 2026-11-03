
from opbench.fla_fixtures import load_fixture
import os
import pytest
import torch
import torch_npu
from yb_db5199ffda6f6e import layer_norm_fwd
from yb_22cded5b91bba3 import compare_data_precision
from yb_22cded5b91bba3 import convert_tensor_with_device_type

@pytest.mark.functiontest
def test_layer_norm_fwd_kernel(load_path):
    data = load_fixture(os.path.join(load_path, 'test_layer_norm_fwd_kernel_cuda.pt'), map_location=torch.device('cpu'), weights_only=False)
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    y, mean, rstd, res_out = layer_norm_fwd(input_data['x'], input_data['weight'], input_data['bias'], input_data['eps'])
    npu_out = {}
    npu_out['y'] = y
    npu_out['mean'] = mean
    npu_out['rstd'] = rstd
    compare_data_precision(data['gpu_output'], npu_out, device_type='cpu')
    print('Test layer_norm_fwd_kernel passed!')
