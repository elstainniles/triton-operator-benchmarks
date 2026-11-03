
from opbench.fla_fixtures import load_fixture
import os
import pytest
import torch
import torch_npu
from yb_c6472eed460398 import rwkv_mix_fwd
from yb_22cded5b91bba3 import compare_data_precision
from yb_22cded5b91bba3 import convert_tensor_with_device_type

@pytest.mark.functiontest
def test_rwkv_seq_mix_kernel(load_dir):
    data = load_fixture(os.path.join(load_dir, 'test_rwkv_seq_mix_kernel_cuda.pt'), map_location=torch.device('cpu'), weights_only=False)
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    x = input_data['x']
    x_prev = input_data['x_prev']
    x_k = input_data['x_k']
    output = rwkv_mix_fwd(x, x_prev, x_k)
    npu_out = {'output': output.cpu()}
    compare_data_precision(data['gpu_output'], npu_out, device_type='cpu')
