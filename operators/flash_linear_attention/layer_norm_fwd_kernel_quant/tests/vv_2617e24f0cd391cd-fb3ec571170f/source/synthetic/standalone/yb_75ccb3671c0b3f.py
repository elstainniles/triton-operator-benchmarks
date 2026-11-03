
from opbench.fla_fixtures import load_fixture
import os
import pytest
import torch
import torch_npu
import triton
from yb_1888053bd1824c import layer_norm_fwd_kernel_quant
from yb_22cded5b91bba3 import compare_data_precision
from yb_22cded5b91bba3 import convert_tensor_with_device_type

@pytest.mark.functiontest
def test_layer_norm_fwd_kernel_quant(load_path):
    data = load_fixture(os.path.join(load_path, 'test_layer_norm_fwd_kernel_quant_cuda.pt'), map_location=torch.device('cpu'), weights_only=False)
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    x = input_data['x']
    weight = input_data['weight']
    bias = input_data['bias']
    M = input_data['M']
    N = input_data['N']
    eps = input_data['eps']
    is_rms_norm = input_data['is_rms_norm']
    y = torch.empty_like(x)
    mean = torch.empty((M,), dtype=torch.float32, device=x.device)
    rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    layer_norm_fwd_kernel_quant[M,](x, y, weight, bias, None, None, mean, rstd, x.stride(0), y.stride(0), 0, 0, N, eps, is_rms_norm, BLOCK_N, False, False, True, True)
    npu_out = {}
    npu_out['y'] = y
    npu_out['mean'] = mean
    npu_out['rstd'] = rstd
    compare_data_precision(data['gpu_output'], npu_out, device_type='cpu')
    print('Test layer_norm_fwd_kernel_quant passed!')
