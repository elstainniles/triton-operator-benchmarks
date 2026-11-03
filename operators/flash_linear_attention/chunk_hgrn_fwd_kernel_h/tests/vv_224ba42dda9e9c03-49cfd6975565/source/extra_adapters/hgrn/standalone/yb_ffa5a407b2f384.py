# Test case for chunk_hgrn_fwd_kernel_h
# Extracted from: generated-data/chunk.py

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

@triton.jit
def exp(x):
    return tl.exp(x.to(tl.float32))
import inspect
SUPPORTS_AUTOTUNE_CACHE = 'cache_results' in inspect.signature(triton.autotune).parameters
FLA_CACHE_RESULTS = os.getenv('FLA_CACHE_RESULTS', '1') == '1'
autotune_cache_kwargs = {'cache_results': FLA_CACHE_RESULTS} if SUPPORTS_AUTOTUNE_CACHE else {}

@triton.autotune(configs=[triton.Config({'BD': 32}, num_warps=1), triton.Config({'BD': 32}, num_warps=2), triton.Config({'BD': 32}, num_warps=4), triton.Config({'BD': 32}, num_warps=8), triton.Config({'BD': 64}, num_warps=1), triton.Config({'BD': 64}, num_warps=2), triton.Config({'BD': 64}, num_warps=4), triton.Config({'BD': 64}, num_warps=8), triton.Config({'BD': 128}, num_warps=1), triton.Config({'BD': 128}, num_warps=2), triton.Config({'BD': 128}, num_warps=4), triton.Config({'BD': 128}, num_warps=8)], key=['D'], **autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def chunk_hgrn_fwd_kernel_h(x, g, gc, o, h0, T, D: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr, USE_INITIAL_STATE: tl.constexpr):
    i_d, i_t, i_b = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    o_d = i_d * BD + tl.arange(0, BD)
    mask = o_d < D
    p_x = x + i_b * T * D + i_t * BT * D + o_d
    p_g = g + i_b * T * D + i_t * BT * D + o_d
    p_gc = gc + i_b * T * D + i_t * BT * D + o_d
    p_o = o + i_b * T * D + i_t * BT * D + o_d
    b_h = tl.zeros([BD], dtype=tl.float32)
    b_gc = tl.zeros([BD], dtype=tl.float32)
    if USE_INITIAL_STATE:
        if i_t == 0:
            b_h += tl.load(h0 + i_b * D + o_d, mask=mask, other=0).to(tl.float32)
    for i in range(0, BT):
        mask_t = mask & (i_t * BT + i < T)
        b_x = tl.load(p_x, mask=mask_t, other=0).to(tl.float32)
        b_g = tl.load(p_g, mask=mask_t, other=0).to(tl.float32)
        b_h = exp(b_g) * b_h + b_x
        b_gc = b_gc + b_g
        tl.store(p_gc, b_gc.to(p_o.dtype.element_ty), mask=mask_t)
        tl.store(p_o, b_h.to(p_o.dtype.element_ty), mask=mask_t)
        p_x += D
        p_g += D
        p_gc += D
        p_o += D

def chunk_hgrn_fwd_h_call(x: torch.Tensor, g: torch.Tensor, initial_state: torch.Tensor=None):
    B, T, D = x.shape
    BT, BD = (128, min(64, triton.next_power_of_2(D)))
    gc = torch.empty_like(g, dtype=torch.float)
    o = torch.empty_like(x, dtype=torch.float)

    def grid(meta):
        return (triton.cdiv(D, meta['BD']), triton.cdiv(T, meta['BT']), B)
    chunk_hgrn_fwd_kernel_h[grid](x, g, gc, o, initial_state, T=T, D=D, BT=BT, USE_INITIAL_STATE=initial_state is not None)
    return (o, gc)

def test_chunk_hgrn_fwd_h():
    B, T, D = (4, 2048, 512)
    print(f'Test Configuration: B={B}, T={T}, D={D}')
    print(f'Input shape: ({B}, {T}, {D})')
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_vv_ops_2/chunk_hgrn_fwd_kernel_h'
    device = torch.device('npu:0')
    x = load_fixture(os.path.join(load_dir, 'x.pt')).to(device)
    g = load_fixture(os.path.join(load_dir, 'g.pt')).to(device)
    h0 = load_fixture(os.path.join(load_dir, 'h0.pt')).to(device)
    print(f'\nInput tensors loaded from {load_dir}')
    print(f'x shape: {x.shape}, dtype: {x.dtype}')
    print(f'g shape: {g.shape}, dtype: {g.dtype}')
    print(f'h0 shape: {h0.shape}, dtype: {h0.dtype}')
    o, gc = chunk_hgrn_fwd_h_call(x=x, g=g, initial_state=h0)
    expected_o = load_fixture(os.path.join(load_dir, 'o.pt')).to(device)
    expected_gc = load_fixture(os.path.join(load_dir, 'gc.pt')).to(device)
    print('Comparing outputs with expected values...')
    validate_cmp('float32', o, expected_o)
    validate_cmp('float32', gc, expected_gc)
    print('✓ Output verification passed!')
    print('\n' + '=' * 80)
    print('Starting Profiling Test for chunk_hgrn_fwd_kernel_h')
    print('=' * 80)
    experimental_config = torch_npu.profiler._ExperimentalConfig(aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization, profiler_level=torch_npu.profiler.ProfilerLevel.Level1, l2_cache=False)
    result_dir = './result_dir_chunk_hgrn_fwd_kernel_h'
    with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.NPU], with_stack=False, record_shapes=False, profile_memory=False, schedule=torch_npu.profiler.schedule(wait=1, warmup=1, active=30, repeat=1, skip_first=1), experimental_config=experimental_config, on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(result_dir)) as prof:
        prof.start()
        for i in range(35):
            o, gc = chunk_hgrn_fwd_h_call(x=x, g=g, initial_state=h0)
            torch.npu.synchronize()
            prof.step()
        prof.stop()
    print(f'\nProfiling completed!')
    print(f'Results saved to: {result_dir}')
    print(f'\nOutput shapes:')
    print(f'o shape: {o.shape}')
    print(f'gc shape: {gc.shape}')
    print('\nTest completed successfully!')
