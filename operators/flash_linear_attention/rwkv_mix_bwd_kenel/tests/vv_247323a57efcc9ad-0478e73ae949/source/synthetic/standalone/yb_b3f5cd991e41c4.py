
from opbench.fla_fixtures import load_fixture
import os
import pytest
import torch
import torch_npu
from yb_c6472eed460398 import rwkv_mix_bwd_kenel
from yb_075f0264c461c9 import compare_data_precision
from yb_075f0264c461c9 import convert_tensor_with_device_type

def rwkv_mix_bwd(dk1: torch.Tensor, x: torch.Tensor, x_prev: torch.Tensor, x_k: torch.Tensor, inplace: bool=True):
    batch_size = x.shape[0] if x.dim() == 3 else 1
    seq_len, hidden_dim = (x.shape[-2], x.shape[-1])
    dk1 = dk1.view(-1, hidden_dim).contiguous()
    dx_prev = torch.empty_like(x_prev) if not inplace else x_prev
    dx = torch.empty_like(x) if not inplace else x

    def grid(meta):
        return ((batch_size * seq_len * hidden_dim + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'], 1, 1)
    rwkv_mix_bwd_kenel[grid](dk1, x_k.squeeze(), dx, dx_prev, batch_size, seq_len, hidden_dim)
    return (dx, dx_prev)

@pytest.mark.functiontest
def test_rwkv_mix_bwd_kenel(load_dir):
    data = load_fixture(os.path.join(load_dir, 'test_rwkv_mix_bwd_kenel_cuda.pt'), map_location=torch.device('cpu'), weights_only=False)
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    x = input_data['x']
    x_prev = input_data['x_prev']
    x_k = input_data['x_k']
    dk1 = input_data['dk1']
    tri_dx, tri_dx_prev = rwkv_mix_bwd(dk1.clone(), x, x_prev, x_k, inplace=False)
    npu_out = {'dx': tri_dx.cpu(), 'dx_prev': tri_dx_prev.cpu()}
    compare_data_precision(data['gpu_output'], npu_out, device_type='cpu')
