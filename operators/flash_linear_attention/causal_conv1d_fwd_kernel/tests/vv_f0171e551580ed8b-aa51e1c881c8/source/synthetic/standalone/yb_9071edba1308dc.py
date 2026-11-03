
from opbench.fla_fixtures import load_fixture
import os
import pytest
import torch
import torch_npu
from yb_ae727972f7e692 import causal_conv1d_fwd
from yb_22cded5b91bba3 import compare_data_precision
from yb_22cded5b91bba3 import convert_tensor_with_device_type

@pytest.mark.functiontest
def test_causal_conv1d_fwd_kernel(load_path):
    data = load_fixture(os.path.join(load_path, 'test_causal_conv1d_fwd_kernel_cuda.pt'), map_location=torch.device('cpu'), weights_only=False)
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    y, _ = causal_conv1d_fwd(input_data['x'], input_data['weight'], input_data['bias'], residual=None)
    npu_out = {}
    npu_out['y'] = y
    compare_data_precision(data['gpu_output'], npu_out, device_type='cpu')
    print('Test causal_conv1d_fwd_kernel passed!')
