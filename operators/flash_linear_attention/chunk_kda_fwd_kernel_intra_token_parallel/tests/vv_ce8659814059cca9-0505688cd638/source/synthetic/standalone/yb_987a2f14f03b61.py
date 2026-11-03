
from opbench.fla_fixtures import load_fixture
import os
import torch
import torch_npu
import pytest
from yb_7b768c9b6799cb import chunk_kda_fwd_intra_token_parallel
from yb_ab7741ad5cca91 import compare_data_precision
from yb_ab7741ad5cca91 import convert_tensor_with_device_type

@pytest.mark.functiontest
def test_chunk_kda_fwd_kernel_intra_token_parallel(load_dir):
    data = load_fixture(os.path.join(load_dir, 'test_chunk_kda_fwd_kernel_intra_token_parallel_cuda.pt'), map_location=torch.device('cpu'), weights_only=False)
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    Aqk_out, Akk_out = chunk_kda_fwd_intra_token_parallel(q=input_data['q'], k=input_data['k'], gk=input_data['gk'], beta=input_data['beta'], Aqk=input_data['Aqk'], Akk=input_data['Akk'], scale=input_data['scale'], chunk_size=input_data['chunk_size'], sub_chunk_size=input_data['sub_chunk_size'])
    npu_out = {'Aqk': Aqk_out.cpu(), 'Akk': Akk_out.cpu()}
    compare_data_precision(data['gpu_output'], npu_out, device_type='cpu')
    print('Testcase Pass')
