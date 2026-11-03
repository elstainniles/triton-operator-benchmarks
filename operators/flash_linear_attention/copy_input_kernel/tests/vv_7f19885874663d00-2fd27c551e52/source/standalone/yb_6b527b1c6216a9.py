
from opbench.fla_fixtures import load_fixture
import os
import pytest
import torch
import torch_npu
from yb_5797d3e04fb6ed import copy_input_kernel
from yb_22cded5b91bba3 import compare_data_precision
from yb_22cded5b91bba3 import convert_tensor_with_device_type

@pytest.mark.functiontest
def test_copy_input_kernel(load_dir):
    data = load_fixture(os.path.join(load_dir, 'test_copy_input_kernel_cuda.pt'), map_location=torch.device('cpu'), weights_only=False)
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    B = input_data['B']
    T = input_data['T']
    H = input_data['H']
    K = input_data['K']
    V = input_data['V']
    L = input_data['L']
    BT = input_data['BT']
    grid = (B * H,)
    copy_input_kernel[grid](q=input_data['q'], k=input_data['k'], v=input_data['v'], g=input_data['g'], level_scales=input_data['level_scales'], cu_seqlens=None, q_prev=input_data['q_prev'], k_prev=input_data['k_prev'], v_prev=input_data['v_prev'], g_prev=input_data['g_prev'], level_scales_prev=input_data['level_scales_prev'], offsets=input_data['offsets'], q_new=input_data['q_new'], k_new=input_data['k_new'], v_new=input_data['v_new'], g_new=input_data['g_new'], level_scales_new=input_data['level_scales_new'], T=T, H=H, K=K, V=V, L=L, BT=BT)
    npu_out = {}
    npu_out['q_new'] = input_data['q_new']
    npu_out['k_new'] = input_data['k_new']
    npu_out['v_new'] = input_data['v_new']
    npu_out['g_new'] = input_data['g_new']
    npu_out['level_scales_new'] = input_data['level_scales_new']
    compare_data_precision(data['gpu_output'], npu_out, device_type='cpu')
    print('Test copy_input_kernel passed!')
