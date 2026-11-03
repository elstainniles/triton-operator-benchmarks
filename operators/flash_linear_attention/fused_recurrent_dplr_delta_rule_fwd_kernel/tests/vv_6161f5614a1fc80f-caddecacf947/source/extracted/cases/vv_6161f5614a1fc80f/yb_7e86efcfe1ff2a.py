
from opbench.fla_fixtures import load_fixture
import os
import pytest
import torch
import torch_npu
from yb_470913040c248c import fused_recurrent_dplr_delta_rule_fwd
from yb_22cded5b91bba3 import compare_data_precision
from yb_22cded5b91bba3 import convert_tensor_with_device_type
os.environ['FLA_USE_FAST_OPS'] = '0'

@pytest.mark.functiontest
def test_fused_recurrent_dplr_delta_rule_fwd_kernel(load_dir):
    data = load_fixture(os.path.join(load_dir, 'test_fused_recurrent_dplr_delta_rule_fwd_kernel_cuda.pt'), map_location=torch.device('cpu'), weights_only=False)
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    o, final_state = fused_recurrent_dplr_delta_rule_fwd(q=input_data['q'], k=input_data['k'], v=input_data['v'], a=input_data['a'], b=input_data['b'], gk=input_data['gk'], scale=input_data['scale'], initial_state=input_data['initial_state'], output_final_state=input_data['output_final_state'], reverse=input_data['reverse'], cu_seqlens=input_data['cu_seqlens'])
    npu_out = {'o': o, 'final_state': final_state}
    compare_data_precision(data['gpu_output'], npu_out, device_type='cpu')
    print('Test fused_recurrent_dplr_delta_rule_fwd_kernel passed!')
