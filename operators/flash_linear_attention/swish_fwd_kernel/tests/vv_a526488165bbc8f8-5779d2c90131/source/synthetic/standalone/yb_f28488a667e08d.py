
from opbench.fla_fixtures import load_fixture
import os
import pytest
import torch
import torch_npu
from yb_dce32337bf0092 import swish_fwd
from yb_22cded5b91bba3 import compare_data_precision
from yb_22cded5b91bba3 import convert_tensor_with_device_type

@pytest.mark.functiontest
def test_swish_fwd_kernel(load_path):
    data = load_fixture(os.path.join(load_path, 'test_swish_fwd_kernel_cuda.pt'), map_location=torch.device('cpu'), weights_only=False)
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    y = swish_fwd(input_data['x'])
    npu_out = {}
    npu_out['y'] = y
    compare_data_precision(data['gpu_output'], npu_out, device_type='cpu')
    print('Test swish_fwd_kernel passed!')
