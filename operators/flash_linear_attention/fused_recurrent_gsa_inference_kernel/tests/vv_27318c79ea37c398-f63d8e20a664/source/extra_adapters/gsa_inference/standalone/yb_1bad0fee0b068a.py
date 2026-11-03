
from opbench.fla_fixtures import load_fixture
import os
import pytest
import torch
import torch_npu
from yb_a77feab5795919 import fused_recurrent_gsa_inference_wrapped
from yb_22cded5b91bba3 import compare_data_precision
from yb_22cded5b91bba3 import convert_tensor_with_device_type

@pytest.mark.functiontest
def test_fused_recurrent_gsa_inference_kernel(load_path):
    data = load_fixture(os.path.join(load_path, 'test_fused_recurrent_gsa_inference_kernel_cuda.pt'), map_location=torch.device('cpu'), weights_only=False)
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    o, (hkt, hvt) = fused_recurrent_gsa_inference_wrapped(input_data['q'], input_data['k'], input_data['v'], input_data['s'], input_data['g'], initial_state=None, output_final_state=input_data['output_final_state'], scale=input_data['scale'])
    npu_out = {}
    npu_out['o'] = o
    npu_out['hkt'] = hkt
    npu_out['hvt'] = hvt
    compare_data_precision(data['gpu_output'], npu_out, device_type='cpu')
    print('Test fused_recurrent_gsa_inference_kernel passed!')
