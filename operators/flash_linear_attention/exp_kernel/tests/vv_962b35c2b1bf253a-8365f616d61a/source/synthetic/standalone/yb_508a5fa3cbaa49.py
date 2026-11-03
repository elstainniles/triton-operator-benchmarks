
from opbench.fla_fixtures import load_fixture
import os
import pytest
import torch
import torch_npu
import triton
import triton.language as tl
from yb_8459c530c3d6be import exp
NPU_ID = os.environ.get('BENCH_NPU_ID', '0')
torch.npu.set_device(int(NPU_ID))
DEVICE = f'npu:{torch.npu.current_device()}'
kernel_name = 'exp'

@triton.jit
def exp_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = exp(x)
    tl.store(output_ptr + offsets, output, mask=mask)

@triton.jit
def baseline_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = x
    tl.store(output_ptr + offsets, output, mask=mask)

def exp_call(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    exp_kernel[grid](x, output, n_elements, BLOCK_SIZE=4096)
    return output

def baseline_call(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    baseline_kernel[grid](x, output, n_elements, BLOCK_SIZE=4096)
    return output

@pytest.mark.functiontest
def test_exp_functiontest(load_dir):
    inputs = load_fixture(os.path.join(load_dir, 'inputs.pt'))
    x = inputs['x'].to(DEVICE)
    output = exp_call(x)
    baseline_output = baseline_call(x)
    torch.npu.synchronize()
    expected_outputs = load_fixture(os.path.join(load_dir, 'outputs.pt'))
    expected_output = expected_outputs['output'].to(DEVICE)
    expected_baseline_output = expected_outputs['baseline_output'].to(DEVICE)
    torch.testing.assert_close(output, expected_output, atol=0.008, rtol=0.008)
    print('output precision check passed')
    torch.testing.assert_close(baseline_output, expected_baseline_output, atol=0.008, rtol=0.008)
    print('baseline_output precision check passed')
    print(f'[{kernel_name}] Precision check completed successfully')
