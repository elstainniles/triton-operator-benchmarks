
from opbench.fla_fixtures import load_fixture
import os
import pytest
import torch
import torch_npu
from yb_176304286c7418 import mean_pooling_bwd
from yb_176304286c7418 import mean_pooling_bwd_kernel
NPU_ID = os.environ.get('BENCH_NPU_ID', '0')
torch.npu.set_device(int(NPU_ID))
DEVICE = f'npu:{torch.npu.current_device()}'
kernel_name = 'mean_pooling_bwd_kernel'

@pytest.mark.functiontest
def test_mean_pooling_bwd_kernel_functiontest(load_dir):
    inputs = load_fixture(os.path.join(load_dir, f'{kernel_name}_in.pt'))
    do = inputs['do'].to(DEVICE)
    B = inputs['B']
    T = inputs['T']
    chunk_size = inputs['chunk_size']
    output = mean_pooling_bwd(do, B, T, chunk_size)
    torch.npu.synchronize()
    expected_outputs = load_fixture(os.path.join(load_dir, f'{kernel_name}_out.pt'))
    expected_output = expected_outputs['output'].to(DEVICE)
    dtype = do.dtype
    if dtype == torch.float32:
        atol, rtol = (1e-05, 1e-05)
    elif dtype == torch.float16:
        atol, rtol = (0.001, 0.001)
    elif dtype == torch.bfloat16:
        atol, rtol = (0.005, 0.005)
    else:
        atol, rtol = (1e-05, 1e-05)
    torch.testing.assert_close(output, expected_output, atol=atol, rtol=rtol)
    print(f'[{kernel_name}] Precision check passed!')
