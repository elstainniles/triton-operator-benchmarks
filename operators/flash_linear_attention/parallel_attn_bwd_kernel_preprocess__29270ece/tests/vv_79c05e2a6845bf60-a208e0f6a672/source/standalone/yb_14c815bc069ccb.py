
from opbench.fla_fixtures import load_fixture
import os
import pytest
import torch
import torch_npu
from yb_29270ece94452f import parallel_attn_bwd_preprocess
from yb_22cded5b91bba3 import compare_data_precision
from yb_22cded5b91bba3 import convert_tensor_with_device_type

@pytest.mark.functiontest
def test_parallel_attn_bwd_kernel_preprocess(load_dir):
    data = load_fixture(os.path.join(load_dir, 'test_parallel_attn_bwd_kernel_preprocess_cuda.pt'), map_location=torch.device('cpu'), weights_only=False)
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    delta = parallel_attn_bwd_preprocess(input_data['o'], input_data['do'])
    npu_out = {}
    npu_out['delta'] = delta
    compare_data_precision(data['gpu_output'], npu_out, device_type='cpu')
    print('Test parallel_attn_bwd_kernel_preprocess passed!')
