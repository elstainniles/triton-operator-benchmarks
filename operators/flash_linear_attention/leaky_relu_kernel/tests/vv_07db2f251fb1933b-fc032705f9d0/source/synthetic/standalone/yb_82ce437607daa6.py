
from opbench.fla_fixtures import load_fixture
import os
import pytest
import torch
import torch_npu
import triton
import triton.language as tl
from yb_74f55558e08fc1 import autotune_cache_kwargs
NPU_ID = os.environ.get('BENCH_NPU_ID', '0')
torch.npu.set_device(int(NPU_ID))
DEVICE = f'npu:{torch.npu.current_device()}'
kernel_name = 'leaky_relu'

@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)

@triton.autotune(configs=[triton.Config({'BLOCK_SIZE': BS}, num_warps=num_warps) for BS in [256, 512, 1024, 2048] for num_warps in [1, 2, 4, 8]], key=['N'], **autotune_cache_kwargs)
@triton.jit
def leaky_relu_kernel(input, output, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(input + offsets, mask=mask, other=0.0).to(tl.float32)
    y = leaky_relu(x)
    tl.store(output + offsets, y.to(output.dtype.element_ty), mask=mask)

def run_leaky_relu(input_tensor):
    N = input_tensor.numel()
    output = torch.empty_like(input_tensor)
    grid = (triton.cdiv(N, 1024),)
    leaky_relu_kernel[grid](input_tensor, output, N)
    return output

@pytest.mark.functiontest
def test_leaky_relu_functiontest(load_dir):
    inputs = load_fixture(os.path.join(load_dir, f'{kernel_name}_in.pt'))
    x = inputs['x'].to(DEVICE)
    output = run_leaky_relu(x)
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
