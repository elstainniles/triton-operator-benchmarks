
from opbench.fla_fixtures import load_fixture
from yb_74f55558e08fc1 import autotune_cache_kwargs
import torch
import torch_npu
import triton
import triton.language as tl
import os
from typing import Optional

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

@triton.jit
def exp(x):
    return tl.exp(x.to(tl.float32))

@triton.jit
def log(x):
    return tl.log(x.to(tl.float32))

def check_shared_mem(arch: str='none', tensor_idx: int=0) -> bool:
    """Check if shared memory meets requirement for given architecture"""
    try:
        import torch
        return True
    except Exception:
        return False

@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None, 'USE_BLOCK_COUNTS': lambda args: isinstance(args['block_counts'], torch.Tensor)})
@triton.autotune(configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4]], key=['BS', 'BK', 'BV'], **autotune_cache_kwargs)
@triton.jit
def parallel_nsa_fwd_kernel(q, k, v, o, lse, scale, block_indices, block_counts, cu_seqlens, token_indices, T, H: tl.constexpr, HQ: tl.constexpr, G: tl.constexpr, K: tl.constexpr, V: tl.constexpr, S: tl.constexpr, BS: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, IS_VARLEN: tl.constexpr, USE_BLOCK_COUNTS: tl.constexpr):
    i_t, i_v, i_bh = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    i_b, i_h = (i_bh // H, i_bh % H)
    if IS_VARLEN:
        i_n, i_t = (tl.load(token_indices + i_t * 2).to(tl.int32), tl.load(token_indices + i_t * 2 + 1).to(tl.int32))
        bos, eos = (tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32))
        T = eos - bos
    else:
        bos, eos = (i_b * T, i_b * T + T)
    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    block_indices += (bos + i_t) * H * S + i_h * S
    if USE_BLOCK_COUNTS:
        NS = tl.load(block_counts + (bos + i_t) * H + i_h)
    else:
        NS = S
    p_q = tl.make_block_ptr(q + (bos + i_t) * HQ * K, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)
    p_o = tl.make_block_ptr(o + (bos + i_t) * HQ * V, (HQ, V), (V, 1), (i_h * G, i_v * BV), (G, BV), (1, 0))
    p_lse = lse + (bos + i_t) * HQ + i_h * G + tl.arange(0, G)
    b_o = tl.zeros([G, BV], dtype=tl.float32)
    b_m = tl.full([G], float('-inf'), dtype=tl.float32)
    b_acc = tl.zeros([G], dtype=tl.float32)
    for i in range(NS):
        i_s = tl.load(block_indices + i).to(tl.int32) * BS
        if i_s <= i_t and i_s >= 0:
            p_k = tl.make_block_ptr(k, (K, T), (1, H * K), (0, i_s), (BK, BS), (0, 1))
            p_v = tl.make_block_ptr(v, (T, V), (H * V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_v = tl.load(p_v, boundary_check=(0, 1))
            b_s = tl.dot(b_q, b_k)
            b_s = tl.where((i_t >= i_s + tl.arange(0, BS))[None, :], b_s, float('-inf'))
            b_m, b_mp = (tl.maximum(b_m, tl.max(b_s, 1)), b_m)
            b_r = exp(b_mp - b_m)
            b_p = exp(b_s - b_m[:, None])
            b_acc = b_acc * b_r + tl.sum(b_p, 1)
            b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_q.dtype), b_v)
            b_mp = b_m
    b_o = b_o / b_acc[:, None]
    b_m += log(b_acc)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_lse, b_m.to(p_lse.dtype.element_ty))

def parallel_nsa_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_indices: torch.LongTensor, block_counts: torch.LongTensor | int, block_size: int, scale: float, cu_seqlens: torch.LongTensor | None=None, token_indices: torch.LongTensor | None=None):
    B, T, H, K, V, S = (*k.shape, v.shape[-1], block_indices.shape[-1])
    HQ = q.shape[2]
    G = HQ // H
    BS = block_size
    if check_shared_mem('hopper', q.device.index):
        BK = min(256, triton.next_power_of_2(K))
        BV = min(256, triton.next_power_of_2(V))
    else:
        BK = min(128, triton.next_power_of_2(K))
        BV = min(128, triton.next_power_of_2(V))
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    assert NK == 1, 'The key dimension can not be larger than 256'
    grid = (T, NV, B * H)
    o = torch.empty(B, T, HQ, V, dtype=v.dtype, device=q.device)
    lse = torch.empty(B, T, HQ, dtype=torch.float, device=q.device)
    parallel_nsa_fwd_kernel[grid](q=q, k=k, v=v, o=o, lse=lse, scale=scale, block_indices=block_indices, block_counts=block_counts, cu_seqlens=cu_seqlens, token_indices=token_indices, T=T, H=H, HQ=HQ, G=G, K=K, V=V, S=S, BS=BS, BK=BK, BV=BV)
    return (o, lse)

def get_test_config():
    """Generate test configuration with large shapes"""
    B = 2
    T = 1024
    H = 8
    HQ = 128
    K = 64
    V = 64
    block_size = 64
    block_counts = 16
    return {'B': B, 'T': T, 'H': H, 'HQ': HQ, 'K': K, 'V': V, 'block_size': block_size, 'block_counts': block_counts}

def test_parallel_nsa_fwd_kernel():
    """Test function for parallel_nsa_fwd_kernel"""
    config = get_test_config()
    B, T, H, HQ, K, V = (config['B'], config['T'], config['H'], config['HQ'], config['K'], config['V'])
    block_size = config['block_size']
    block_counts = config['block_counts']
    device = torch.device('npu:0')
    dtype = torch.float16
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/parallel_nsa_fwd_kernel/'
    q = load_fixture(os.path.join(load_dir, 'q.pt')).to(device)
    k = load_fixture(os.path.join(load_dir, 'k.pt')).to(device)
    v = load_fixture(os.path.join(load_dir, 'v.pt')).to(device)
    block_indices = load_fixture(os.path.join(load_dir, 'block_indices.pt')).to(device)
    print(f'Input tensors loaded from {load_dir}')
    scale = K ** (-0.5)
    print(f'Test configuration:')
    print(f'  Batch size (B): {B}')
    print(f'  Sequence length (T): {T}')
    print(f'  Key/Value heads (H): {H}')
    print(f'  Query heads (HQ): {HQ}')
    print(f'  Key dimension (K): {K}')
    print(f'  Value dimension (V): {V}')
    print(f'  Block size: {block_size}')
    print(f'  Block counts: {block_counts}')
    print(f'  Query shape: {q.shape}')
    print(f'  Key shape: {k.shape}')
    print(f'  Value shape: {v.shape}')
    print(f'  Block indices shape: {block_indices.shape}')
    o, lse = parallel_nsa_fwd(q=q, k=k, v=v, block_indices=block_indices, block_counts=block_counts, block_size=block_size, scale=scale, cu_seqlens=None, token_indices=None)
    o_expected = load_fixture(os.path.join(load_dir, 'o.pt')).to(device)
    lse_expected = load_fixture(os.path.join(load_dir, 'lse.pt')).to(device)
    print(f'Expected output tensors loaded from {load_dir}')
    print(f'\nComparing outputs with expected values...')
    validate_cmp('float16', o, o_expected)
    validate_cmp('float16', lse, lse_expected)
    print(f'Output comparison passed!')
    print(f'\nOutput shape: {o.shape}')
    print(f'Output dtype: {o.dtype}')
    print(f'LSE shape: {lse.shape}')
    print(f'Sample output values (first 5): {o[0, 0, 0, :5]}')
    print(f'\nTest passed!')
    return (o, lse)
