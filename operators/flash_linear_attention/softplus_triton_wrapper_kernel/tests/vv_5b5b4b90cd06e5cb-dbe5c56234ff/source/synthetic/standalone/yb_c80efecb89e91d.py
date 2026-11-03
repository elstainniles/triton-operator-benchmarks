
from opbench.fla_fixtures import load_fixture
import os
import pytest
import torch
import torch_npu
import triton
import triton.language as tl
from yb_90891058742c80 import softplus_triton
NPU_ID = os.environ.get('BENCH_NPU_ID', '0')
torch.npu.set_device(int(NPU_ID))
DEVICE = f'npu:{torch.npu.current_device()}'
N = 1024 * 1024
kernel_name = 'softplus_triton'

@triton.jit
def softplus_triton_wrapper_kernel(x_ptr, y_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = softplus_triton(x)
    tl.store(y_ptr + offsets, y, mask=mask)

def call_func(x, y):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    softplus_triton_wrapper_kernel[grid](x, y, N, BLOCK_SIZE)
    return y

@pytest.mark.functiontest
def test_softplus_triton_functiontest(load_dir):
    inputs = load_fixture(os.path.join(load_dir, f'{kernel_name}_in.pt'))
    x = inputs['x'].to(DEVICE)
    y = torch.empty_like(x)
    output = call_func(x, y)
    torch.npu.synchronize()
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
    print(f'Precision check passed for {kernel_name}')
    print(f'[{kernel_name}] Precision check completed successfully')
