
from opbench.fla_fixtures import load_fixture
import pytest
import torch
import torch_npu
from yb_55d32324cd9df0 import chunk_rwkv6_fwd_intra_sub_intra_split
from yb_22cded5b91bba3 import compare_data_precision
from yb_22cded5b91bba3 import convert_tensor_with_device_type

@pytest.mark.functiontest
def test_chunk_rwkv6_fwd_A_kernel_intra_sub_intra_split():
    data = load_fixture('/data/PytorchFile/Inference_or_Training_solution/fla_vv_ops_2/test_chunk_rwkv6_fwd_A_kernel_intra_sub_intra_split/test_chunk_rwkv6_fwd_A_kernel_intra_sub_intra_split_cuda.pt', map_location=torch.device('cpu'), weights_only=False)
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    A_intra = chunk_rwkv6_fwd_intra_sub_intra_split(input_data['q'], input_data['k'], input_data['gi'], input_data['ge'], input_data['u'], scale=input_data['scale'], chunk_size=input_data['chunk_size'])
    npu_out = {}
    npu_out['A_intra'] = A_intra
    compare_data_precision(data['gpu_output'], npu_out, device_type='cpu')
    print('Test chunk_rwkv6_fwd_A_kernel_intra_sub_intra_split passed!')
