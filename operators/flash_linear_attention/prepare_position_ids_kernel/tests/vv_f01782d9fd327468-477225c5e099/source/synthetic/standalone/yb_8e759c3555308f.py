
from opbench.fla_fixtures import load_fixture
import os
import pytest
import torch
import torch_npu
from yb_4f3456615af619 import prepare_position_ids_kernel
NPU_ID = os.environ.get('BENCH_NPU_ID', '0')
torch.npu.set_device(int(NPU_ID))
DEVICE = f'npu:{torch.npu.current_device()}'
kernel_name = 'prepare_position_ids_kernel'

def _get_value(val):
    return val.item() if isinstance(val, torch.Tensor) else val

@pytest.mark.functiontest
def test_prepare_position_ids_kernel_functiontest(load_dir):
    inputs = load_fixture(os.path.join(load_dir, f'{kernel_name}_in.pt'))
    y = inputs['y'].to(DEVICE)
    cu_seqlens = inputs['cu_seqlens'].to(DEVICE)
    B_KERNEL = _get_value(inputs['B_KERNEL'])
    B = _get_value(inputs['B'])
    grid = (B,)
    prepare_position_ids_kernel[grid](y, cu_seqlens, B_KERNEL)
    expected_outputs = load_fixture(os.path.join(load_dir, f'{kernel_name}_out.pt'))
    expected_output = expected_outputs['output'].to(DEVICE)
    dtype = y.dtype
    if dtype == torch.float32:
        atol, rtol = (1e-05, 1e-05)
    elif dtype == torch.float16:
        atol, rtol = (0.001, 0.001)
    elif dtype == torch.bfloat16:
        atol, rtol = (0.005, 0.005)
    else:
        atol, rtol = (1e-05, 1e-05)
    torch.testing.assert_close(y, expected_output, atol=atol, rtol=rtol)
    print('Output precision check passed')
    print(f'[{kernel_name}] Precision check completed successfully')
