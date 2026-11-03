
from opbench.fla_fixtures import load_fixture
import os
import pytest
import torch
import torch_npu
NPU_ID = os.environ.get('BENCH_NPU_ID', '0')
torch.npu.set_device(int(NPU_ID))
DEVICE = f'npu:{torch.npu.current_device()}'
kernel_name = 'compute_dh0_kernel'

@pytest.mark.functiontest
def test_compute_dh0_kernel_functiontest(load_dir):
    from yb_717c6a81dac3a7 import compute_dh0_triton
    inputs = load_fixture(os.path.join(load_dir, f'{kernel_name}_in.pt'))
    dy = inputs['dy'].to(DEVICE)
    y = inputs['y'].to(DEVICE)
    weight = inputs['weight'].to(DEVICE)
    initial_state = inputs['initial_state'].to(DEVICE)
    activation = inputs['activation']
    seq_idx = inputs['seq_idx']
    dh0 = compute_dh0_triton(dy, y, weight, initial_state, activation, seq_idx)
    torch.npu.synchronize()
    expected_outputs = load_fixture(os.path.join(load_dir, f'{kernel_name}_out.pt'))
    expected_dh0 = expected_outputs['dh0'].to(DEVICE)
    dtype = dy.dtype
    if dtype == torch.float32:
        atol, rtol = (1e-05, 1e-05)
    elif dtype == torch.float16:
        atol, rtol = (0.001, 0.001)
    elif dtype == torch.bfloat16:
        atol, rtol = (0.005, 0.005)
    else:
        atol, rtol = (1e-05, 1e-05)
    torch.testing.assert_close(dh0, expected_dh0, atol=atol, rtol=rtol)
    print(f'Precision check passed for {kernel_name}')
