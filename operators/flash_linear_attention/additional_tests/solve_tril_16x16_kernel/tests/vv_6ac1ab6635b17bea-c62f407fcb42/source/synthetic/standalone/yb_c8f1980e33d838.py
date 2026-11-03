
from opbench.fla_fixtures import load_fixture
import os
import pytest
import torch
import torch_npu
from yb_35b249daa9eeeb import solve_tril
NPU_ID = os.environ.get('BENCH_NPU_ID', '0')
torch.npu.set_device(int(NPU_ID))
DEVICE = f'npu:{torch.npu.current_device()}'
kernel_name = 'solve_tril_16x16_kernel'

@pytest.mark.functiontest
def test_solve_tril_16x16_kernel_functiontest(load_dir):
    inputs = load_fixture(os.path.join(load_dir, f'{kernel_name}_in.pt'))
    A_input = inputs['A_input'].to(DEVICE)
    B = inputs['B']
    T = inputs['T']
    H = inputs['H']
    BT = inputs['BT']
    output = solve_tril(A_input)
    torch.npu.synchronize()
    expected_outputs = load_fixture(os.path.join(load_dir, f'{kernel_name}_out.pt'))
    expected_output = expected_outputs['output'].to(DEVICE)
    dtype = A_input.dtype
    if dtype == torch.float32:
        atol, rtol = (1e-05, 1e-05)
    elif dtype == torch.float16:
        atol, rtol = (0.001, 0.001)
    elif dtype == torch.bfloat16:
        atol, rtol = (0.005, 0.005)
    else:
        atol, rtol = (1e-05, 1e-05)
    torch.testing.assert_close(output, expected_output, atol=atol, rtol=rtol)
    print(f'[{kernel_name}] Precision check passed')
