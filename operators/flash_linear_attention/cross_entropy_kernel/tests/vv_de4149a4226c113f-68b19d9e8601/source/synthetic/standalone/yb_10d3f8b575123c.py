
from opbench.fla_fixtures import load_fixture
import os
import pytest
import torch
import torch_npu
from yb_0c04dfbe64c0ce import logsumexp_fwd
from yb_0c04dfbe64c0ce import cross_entropy_kernel
from yb_22cded5b91bba3 import compare_data_precision
from yb_22cded5b91bba3 import convert_tensor_with_device_type

@pytest.mark.functiontest
def test_cross_entropy_kernel(load_path):
    data = load_fixture(os.path.join(load_path, 'test_cross_entropy_kernel_cuda.pt'), map_location=torch.device('cpu'), weights_only=False)
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    logits = input_data['logits']
    lse = input_data['lse']
    target = input_data['target']
    B = input_data['B']
    T = input_data['T']
    V = input_data['V']
    loss = torch.empty(B, T, dtype=torch.float, device=logits.device)
    grid = (B * T,)
    BV = 128
    cross_entropy_kernel[grid](logits, lse, target, loss, input_data['total'], input_data['ignore_index'], input_data['label_smoothing'], input_data['logit_scale'], input_data['logit_softcapping'], input_data['reduction'], V, BV)
    npu_out = {}
    npu_out['loss'] = loss
    compare_data_precision(data['gpu_output'], npu_out, device_type='cpu')
    print('Test cross_entropy_kernel passed!')
