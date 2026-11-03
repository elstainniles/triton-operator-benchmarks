
from opbench.fla_fixtures import load_fixture
import os
import pytest
import torch
import torch_npu
from yb_c6472eed460398 import relu_square_bwd_kernel
from yb_22cded5b91bba3 import compare_data_precision
from yb_22cded5b91bba3 import convert_tensor_with_device_type

def relu_square_bwd(grad_output: torch.Tensor, forward_input: torch.Tensor, inplace: bool=True):
    grad_output = grad_output.contiguous()
    output = grad_output if inplace else grad_output.clone()
    forward_input = forward_input.contiguous()
    BLOCK_SIZE = 4096
    grid = ((output.numel() + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    relu_square_bwd_kernel[grid](output, forward_input, BLOCK_SIZE=BLOCK_SIZE)
    return output

@pytest.mark.functiontest
def test_relu_square_bwd_kernel(load_dir):
    data = load_fixture(os.path.join(load_dir, 'test_relu_square_bwd_kernel_cuda.pt'), map_location=torch.device('cpu'), weights_only=False)
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    output = relu_square_bwd(input_data['grad_output'].clone(), input_data['forward_input'], inplace=False)
    npu_out = {'output': output.cpu()}
    compare_data_precision(data['gpu_output'], npu_out, device_type='cpu')
