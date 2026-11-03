
from opbench.fla_fixtures import load_fixture
import os
import pytest
import torch
import torch_npu
from yb_c8f34344490fdd import fused_recurrent_rwkv6_bwd_dw
from yb_f95f816dd5c2a2 import compare_data_precision
from yb_f95f816dd5c2a2 import convert_tensor_with_device_type

@pytest.mark.functiontest
def test_fused_recurrent_rwkv6_bwd_kernel_dw():
    load_path = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/test_fused_recurrent_rwkv6_bwd_kernel_dw/'
    data = load_fixture(os.path.join(load_path, 'test_fused_recurrent_rwkv6_bwd_kernel_dw_cuda.pt'), map_location=torch.device('cpu'), weights_only=False)
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    dw = fused_recurrent_rwkv6_bwd_dw(input_data['q'], input_data['k'], input_data['dq1'], input_data['dk1'], scale=input_data['scale'], reverse=input_data['reverse'])
    npu_out = {}
    npu_out['dw'] = dw
    compare_data_precision(data['gpu_output'], npu_out, device_type='cpu')
    print('Test fused_recurrent_rwkv6_bwd_kernel_dw passed!')
