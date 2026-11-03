
from opbench.fla_fixtures import load_fixture
import os
import pytest
import torch
import torch_npu
from yb_5e473bf2574430 import fused_recurrent_rwkv6_fwd
from yb_22cded5b91bba3 import compare_data_precision
from yb_22cded5b91bba3 import convert_tensor_with_device_type

@pytest.mark.functiontest
def test_fused_recurrent_rwkv6_fwd_kernel():
    data = load_fixture(os.path.join('/data/PytorchFile/Inference_or_Training_solution/fla_vv_ops_2/test_fuesd_recurrent_rwkv6_fwd_kernel', 'test_fused_recurrent_rwkv6_fwd_kernel_cuda.pt'), map_location=torch.device('cpu'), weights_only=False)
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    o, ht = fused_recurrent_rwkv6_fwd(input_data['q'], input_data['k'], input_data['v'], input_data['w'], input_data['u'], scale=input_data['scale'], initial_state=input_data['h0'], output_final_state=input_data['output_final_state'], reverse=input_data['reverse'])
    npu_out = {}
    npu_out['o'] = o
    npu_out['ht'] = ht
    compare_data_precision(data['gpu_output'], npu_out, device_type='cpu')
    print('Test fused_recurrent_rwkv6_fwd_kernel passed!')
