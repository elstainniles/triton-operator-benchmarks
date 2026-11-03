
from opbench.fla_fixtures import load_fixture
import os
import pytest
import torch
import torch_npu
from yb_282df376bcfe75 import chunk_rwkv6_fwd_intra_sub_inter
from yb_f95f816dd5c2a2 import compare_data_precision
from yb_f95f816dd5c2a2 import convert_tensor_with_device_type

@pytest.mark.functiontest
def test_chunk_rwkv6_fwd_A_kernel_intra_sub_inter():
    load_path = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/test_chunk_rwkv6_fwd_A_kernel_intra_sub_inter'
    data = load_fixture(os.path.join(load_path, 'test_chunk_rwkv6_fwd_A_kernel_intra_sub_inter_cuda.pt'), map_location=torch.device('cpu'), weights_only=False)
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    A = chunk_rwkv6_fwd_intra_sub_inter(input_data['q'], input_data['k'], input_data['gi'], input_data['ge'], scale=input_data['scale'], chunk_size=input_data['chunk_size'])
    npu_out = {}
    npu_out['A'] = A
    compare_data_precision(data['gpu_output'], npu_out, device_type='cpu')
    print('Test chunk_rwkv6_fwd_A_kernel_intra_sub_inter passed!')
