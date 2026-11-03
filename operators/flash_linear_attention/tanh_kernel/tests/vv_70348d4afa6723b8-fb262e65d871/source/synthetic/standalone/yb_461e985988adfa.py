
from opbench.fla_fixtures import load_fixture
import os
import pytest
import torch
import torch_npu
import triton
import triton.language as tl
import triton.language.extra.cann.libdevice as tldevice
os.environ['FLA_USE_FAST_OPS'] = '0'
NPU_ID = os.environ.get('BENCH_NPU_ID', '0')
torch.npu.set_device(int(NPU_ID))
DEVICE = f'npu:{torch.npu.current_device()}'
B, T, H, D = (2, 1024, 8, 1024)
kernel_name = 'tanh'

@triton.jit
def tanh(x):
    return tldevice.tanh(x.to(tl.float32))

@triton.jit
def tanh_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tanh(x)
    tl.store(y_ptr + offsets, y, mask=mask)

def tanh_call(x: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    tanh_kernel[grid](x, y, n_elements, BLOCK_SIZE=1024)
    return y

@pytest.mark.functiontest
def test_tanh_kernel_functiontest(load_dir):
    inputs = load_fixture(os.path.join(load_dir, f'{kernel_name}_in.pt'))
    x = inputs['x'].to(DEVICE)
    output = tanh_call(x)
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
    print(f'[{kernel_name}] B={B}, T={T}, H={H}, D={D}, dtype={dtype}, precision check PASSED (atol={atol}, rtol={rtol})')
