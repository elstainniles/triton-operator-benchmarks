
from opbench.fla_fixtures import load_fixture
import os
import pytest
import torch
import torch_npu
from yb_712432704460d6 import chunk_local_cumsum_scalar
NPU_ID = os.environ.get('BENCH_NPU_ID', '0')
torch.npu.set_device(int(NPU_ID))
DEVICE = f'npu:{torch.npu.current_device()}'
kernel_name = 'chunk_local_cumsum_scalar_kernel'

def _get_value(val):
    return val.item() if isinstance(val, torch.Tensor) else val

@pytest.mark.functiontest
def test_chunk_local_cumsum_scalar_kernel_functiontest(load_dir):
    inputs = load_fixture(os.path.join(load_dir, f'{kernel_name}_in.pt'))
    g = inputs['g'].to(DEVICE)
    cs = _get_value(inputs['chunk_size'])
    output = chunk_local_cumsum_scalar(g, cs)
    dtype = g.dtype
    if dtype == torch.float32:
        atol, rtol = (1e-05, 1e-05)
    elif dtype == torch.float16:
        atol, rtol = (0.001, 0.001)
    elif dtype == torch.bfloat16:
        atol, rtol = (0.005, 0.005)
    else:
        atol, rtol = (1e-05, 1e-05)
    expected_outputs = load_fixture(os.path.join(load_dir, f'{kernel_name}_out.pt'))
    if isinstance(output, tuple):
        for i, out in enumerate(output):
            expected_out = expected_outputs[f'output_{i}'].to(DEVICE)
            torch.testing.assert_close(out, expected_out, atol=atol, rtol=rtol)
            print(f'Output {i} precision check passed')
    else:
        expected_output = expected_outputs['output'].to(DEVICE)
        torch.testing.assert_close(output, expected_output, atol=atol, rtol=rtol)
        print('Output precision check passed')
    print(f'[{kernel_name}] Precision check completed successfully')
