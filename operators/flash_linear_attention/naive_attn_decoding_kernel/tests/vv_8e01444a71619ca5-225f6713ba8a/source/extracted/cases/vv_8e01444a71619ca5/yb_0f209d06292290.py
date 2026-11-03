
from opbench.fla_fixtures import load_fixture
import os
import pytest
import torch
import torch_npu
from yb_03aaadc1e61d1b import attn_decoding_one_step
from yb_22cded5b91bba3 import compare_data_precision
from yb_22cded5b91bba3 import convert_tensor_with_device_type

@pytest.mark.functiontest
def test_naive_attn_decoding_kernel(load_dir):
    data = load_fixture(os.path.join(load_dir, 'test_naive_attn_decoding_kernel_cuda.pt'), map_location=torch.device('cpu'), weights_only=False)
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    o = attn_decoding_one_step(input_data['q'], input_data['k'], input_data['v'], g=input_data['g'], scale=input_data['scale'], cu_seqlens=input_data['cu_seqlens'])
    npu_out = {'o': o}
    compare_data_precision(data['gpu_output'], npu_out, device_type='cpu')
    print('Test naive_attn_decoding_kernel passed!')
