
from opbench.fla_fixtures import load_fixture
import os
import pytest
import torch
import torch_npu
from yb_aef642efa9b1c6 import softmax_fwd
NPU_ID = os.environ.get('BENCH_NPU_ID', '0')
torch.npu.set_device(int(NPU_ID))
DEVICE = f'npu:{torch.npu.current_device()}'
kernel_name = 'softmax_fwd_kernel'

@pytest.mark.functiontest
def test_softmax_fwd_kernel_functiontest(load_dir):
    inputs = load_fixture(os.path.join(load_dir, f'{kernel_name}_in.pt'))
    x = inputs['x'].to(DEVICE)
    output = softmax_fwd(x)
    expected_outputs = load_fixture(os.path.join(load_dir, f'{kernel_name}_out.pt'))
    expected_output = expected_outputs['output'].to(DEVICE)
    dtype = x.dtype
    if dtype == torch.float32:
        atol, rtol = (1e-05, 1e-05)
    elif dtype == torch.float16:
        atol, rtol = (0.001, 0.001)
    elif dtype == torch.bfloat16:
        atol, rtol = (0.005, 0.005)
    else:
        atol, rtol = (1e-05, 1e-05)
    torch.testing.assert_close(output, expected_output, atol=atol, rtol=rtol)
    print('Output precision check passed')
    print(f'[{kernel_name}] Precision check completed successfully')
