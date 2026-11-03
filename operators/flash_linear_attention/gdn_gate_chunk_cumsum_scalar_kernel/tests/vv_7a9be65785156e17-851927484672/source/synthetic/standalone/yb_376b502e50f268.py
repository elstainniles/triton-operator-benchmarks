
from opbench.fla_fixtures import load_fixture
import os
import torch
import torch_npu
import pytest
from yb_aa8d11f8979997 import gdn_gate_chunk_cumsum
from yb_22cded5b91bba3 import compare_data_precision
from yb_22cded5b91bba3 import convert_tensor_with_device_type

@pytest.mark.functiontest
def test_gdn_gate_chunk_cumsum_scalar_kernel(load_dir):
    data = load_fixture(os.path.join(load_dir, 'test_gdn_gate_chunk_cumsum_scalar_kernel_cuda.pt'), map_location=torch.device('cpu'), weights_only=False)
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    o = gdn_gate_chunk_cumsum(g=input_data['g'].clone(), A_log=input_data['A_log'].clone(), chunk_size=input_data['chunk_size'], scale=input_data['scale'], dt_bias=input_data['dt_bias'].clone())
    npu_out = {'o': o.cpu()}
    compare_data_precision(data['gpu_output'], npu_out, device_type='cpu')
