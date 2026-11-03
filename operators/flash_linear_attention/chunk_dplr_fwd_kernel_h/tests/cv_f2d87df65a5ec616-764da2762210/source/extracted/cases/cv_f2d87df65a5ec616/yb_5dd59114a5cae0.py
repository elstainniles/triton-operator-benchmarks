
from opbench.fla_fixtures import load_fixture
import os
os.environ['FLA_USE_FAST_OPS'] = '0'
import torch
import torch_npu
from yb_f95f816dd5c2a2 import compare_data_precision
from yb_f95f816dd5c2a2 import convert_tensor_with_device_type
from yb_d2bfc9bec7ae1c import chunk_dplr_fwd_h

def test_chunk_dplr_fwd_kernel_h():
    ptfile_path = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_dplr_fwd_kernel_h/test_chunk_dplr_fwd_kernel_h_cuda.pt'
    data = load_fixture(ptfile_path, map_location=torch.device('cpu'), weights_only=False)
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    h, v_new, final_state = chunk_dplr_fwd_h(kg=input_data['kg'], v=input_data['v'], w=input_data['w'], u=input_data['u'], bg=input_data['bg'], gk=input_data['gk'], initial_state=input_data['initial_state'], output_final_state=input_data['output_final_state'], cu_seqlens=input_data['cu_seqlens'], chunk_size=input_data['chunk_size'], chunk_indices=input_data['chunk_indices'])
    npu_out = {}
    npu_out['h'] = h
    npu_out['v_new'] = v_new
    npu_out['final_state'] = final_state
    compare_data_precision(data['gpu_output'], npu_out, device_type='cpu')
    print('Test chunk_dplr_fwd_kernel_h passed!')
