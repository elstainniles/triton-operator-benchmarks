
from opbench.fla_fixtures import load_fixture
import os
import pytest
import torch
import torch_npu
from yb_de4b66b89bc795 import logsumexp_fwd
from yb_22cded5b91bba3 import compare_data_precision
from yb_22cded5b91bba3 import convert_tensor_with_device_type

@pytest.mark.functiontest
def test_logsumexp_fwd_kernel(load_path):
    data = load_fixture(os.path.join(load_path, 'test_logsumexp_fwd_kernel_cuda.pt'), map_location=torch.device('cpu'), weights_only=False)
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    z = logsumexp_fwd(input_data['x'], scale=input_data['scale'])
    npu_out = {}
    npu_out['z'] = z
    compare_data_precision(data['gpu_output'], npu_out, device_type='cpu')
    print('Test logsumexp_fwd_kernel passed!')
