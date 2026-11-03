# Test case for parallel_nsa_kernel_mask
# Extracted from generated-data/parallel.py

from opbench.fla_fixtures import load_fixture
from typing import Optional
import torch
import torch_npu
import triton
import triton.language as tl
import os

def validate_cmp(dtype, y_cal, y_ref, overflow_mode: Optional[str]=None, device_type: Optional[str]=None):
    if device_type is not None:
        target_device = torch.device(device_type)
        y_cal = y_cal.to(target_device)
        y_ref = y_ref.to(target_device)
    else:
        y_cal = y_cal.npu()
        y_ref = y_ref.npu()
    if overflow_mode == 'saturate':
        if dtype in ['float32', 'float16']:
            min_value = -torch.finfo(dtype).min
            max_value = torch.finfo(dtype).max
        elif dtype in ['int32', 'int16', 'int8']:
            min_value = torch.iinfo(dtype).min
            max_value = torch.iinfo(dtype).max
        elif dtype == 'bool':
            min_value = 0
            max_value = 1
        else:
            raise ValueError('Invalid parameter "dtype" is found : {}'.format(dtype))
        y_ref = torch.clamp(y_ref, min=min_value, max=max_value)
    if dtype == 'float16':
        torch.testing.assert_close(y_ref, y_cal, rtol=0.001, atol=0.001, equal_nan=True)
    elif dtype == 'bfloat16':
        torch.testing.assert_close(y_ref.to(torch.float32), y_cal.to(torch.float32), rtol=0.005, atol=0.005, equal_nan=True)
    elif dtype == 'float32':
        torch.testing.assert_close(y_ref, y_cal, rtol=1e-05, atol=1e-05, equal_nan=True)
    elif dtype == 'int32' or dtype == 'int64' or dtype == 'int16' or (dtype == 'int8'):
        assert torch.equal(y_cal, y_ref)
    elif dtype == 'bool':
        assert torch.equal(y_cal, y_ref)
    else:
        raise ValueError('Invalid parameter "dtype" is found : {}'.format(dtype))

@triton.heuristics({'USE_BLOCK_COUNTS': lambda args: isinstance(args['block_counts'], torch.Tensor)})
@triton.jit(do_not_specialize=['T'])
def parallel_nsa_kernel_mask(block_indices, block_counts, block_mask, T, H: tl.constexpr, S: tl.constexpr, BS: tl.constexpr, NS: tl.constexpr, USE_BLOCK_COUNTS: tl.constexpr):
    i_t, i_b, i_hs = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    i_h, i_s = (i_hs // S, i_hs % S)
    b_i = tl.load(block_indices + i_b * T * H * S + i_t * H * S + i_h * S + i_s)
    if USE_BLOCK_COUNTS:
        b_m = b_i * BS <= i_t and i_s < tl.load(block_counts + i_b * T * H + i_t * H + i_h)
    else:
        b_m = b_i * BS <= i_t
    if b_i < NS and b_i >= 0:
        tl.store(block_mask + i_b * T * H * NS + i_t * H * NS + i_h * NS + b_i, b_m.to(block_mask.dtype.element_ty))

def parallel_nsa_block_mask(block_indices: torch.LongTensor, block_counts: torch.LongTensor | int, cu_seqlens: torch.LongTensor, block_size: int):
    B, T, H, S = block_indices.shape
    BS = block_size
    if cu_seqlens is not None:
        NS = triton.cdiv(T, BS)
    else:
        NS = triton.cdiv(T, BS)
    block_mask = torch.zeros(B, T, H, NS, dtype=torch.bool, device=block_indices.device)
    parallel_nsa_kernel_mask[T, B, H * S](block_indices=block_indices, block_counts=block_counts, block_mask=block_mask, T=T, H=H, S=S, BS=BS, NS=NS)
    return block_mask

def get_test_config():
    """Generate test configuration with large shapes"""
    B = 1
    T = 512
    H = 4
    block_size = 64
    block_counts = 8
    return {'B': B, 'T': T, 'H': H, 'block_size': block_size, 'block_counts': block_counts}

def test_parallel_nsa_kernel_mask():
    """Test function for parallel_nsa_kernel_mask"""
    config = get_test_config()
    B, T, H = (config['B'], config['T'], config['H'])
    block_size = config['block_size']
    block_counts = config['block_counts']
    device = torch.device('npu:0')
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_vv_ops_2/parallel_nsa_kernel_mask'
    block_indices = load_fixture(os.path.join(load_dir, 'block_indices.pt')).to(device)
    print(f'Input tensors loaded from {load_dir}')
    print(f'Test configuration:')
    print(f'  Batch size (B): {B}')
    print(f'  Sequence length (T): {T}')
    print(f'  Number of heads (H): {H}')
    print(f'  Block size: {block_size}')
    print(f'  Block counts: {block_counts}')
    print(f'  Block indices shape: {block_indices.shape}')
    block_mask = parallel_nsa_block_mask(block_indices=block_indices, block_counts=block_counts, cu_seqlens=None, block_size=block_size)
    block_mask_expected = load_fixture(os.path.join(load_dir, 'block_mask.pt')).to(device)
    print(f'Expected output tensors loaded from {load_dir}')
    print(f'\nComparing outputs with expected values...')
    validate_cmp('int32', block_mask, block_mask_expected)
    print(f'Output comparison passed!')
    print(f'\nOutput block_mask shape: {block_mask.shape}')
    print(f'Output dtype: {block_mask.dtype}')
    print(f'Sample block mask values: {block_mask[0, 0, 0, :5]}')
    print(f'\nTest passed!')
    return block_mask
