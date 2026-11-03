
from opbench.fla_fixtures import load_fixture
import os
import pytest
import torch
import torch_npu
from yb_c506fc7ce61fac import fused_recurrent_rwkv7_fwd
from yb_22cded5b91bba3 import compare_data_precision
from yb_22cded5b91bba3 import convert_tensor_with_device_type

@pytest.mark.functiontest
def test_fused_recurrent_rwkv7_fwd_kernel(load_dir):
    data = load_fixture(os.path.join(load_dir, 'test_fused_recurrent_rwkv7_fwd_kernel_cuda.pt'), map_location=torch.device('cpu'), weights_only=False)
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    o, ht = fused_recurrent_rwkv7_fwd(input_data['r'], input_data['w'], input_data['k'], input_data['v'], input_data['kk'], input_data['a'], input_data['scale'], initial_state=input_data['initial_state'], output_final_state=input_data['output_final_state'], reverse=input_data['reverse'], cu_seqlens=input_data['cu_seqlens'])
    npu_out = {'o': o, 'ht': ht}
    compare_data_precision(data['gpu_output'], npu_out, device_type='cpu')
    print('Test fused_recurrent_rwkv7_fwd_kernel passed!')
