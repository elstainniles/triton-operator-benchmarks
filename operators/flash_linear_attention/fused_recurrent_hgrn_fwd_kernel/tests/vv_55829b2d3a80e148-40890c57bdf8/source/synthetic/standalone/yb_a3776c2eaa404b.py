# Test case for fused_recurrent_hgrn_fwd_kernel
# Extracted from: generated-data/fused_recurrent.py

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

@triton.heuristics({'USE_INITIAL_STATE': lambda args: args['h0'] is not None, 'STORE_FINAL_STATE': lambda args: args['ht'] is not None, 'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({'BD': BD}, num_warps=num_warps) for BD in [32, 64, 128] for num_warps in [1, 2, 4, 8]], key=['D'], **autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_hgrn_fwd_kernel(x, g, o, h0, ht, cu_seqlens, T, D: tl.constexpr, BD: tl.constexpr, USE_INITIAL_STATE: tl.constexpr, STORE_FINAL_STATE: tl.constexpr, IS_VARLEN: tl.constexpr):
    i_d, i_n = (tl.program_id(0), tl.program_id(1))
    if IS_VARLEN:
        bos, eos = (tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64))
        T = eos - bos
    else:
        bos, eos = (i_n * T, i_n * T + T)
    o_d = i_d * BD + tl.arange(0, BD)
    mask = o_d < D
    p_x = x + bos * D + o_d
    p_g = g + bos * D + o_d
    p_o = o + bos * D + o_d
    b_h = tl.zeros([BD], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_n * D + o_d
        b_h += tl.load(p_h0, mask=mask, other=0).to(tl.float32)
    for _ in range(0, T):
        b_x = tl.load(p_x, mask=mask, other=0).to(tl.float32)
        b_g = tl.load(p_g, mask=mask, other=0).to(tl.float32)
        b_h = exp(b_g) * b_h + b_x
        tl.store(p_o, b_h.to(p_o.dtype.element_ty), mask=mask)
        p_x += D
        p_g += D
        p_o += D
    if STORE_FINAL_STATE:
        p_ht = ht + i_n * D + o_d
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask)

def fused_recurrent_hgrn_fwd(x: torch.Tensor, g: torch.Tensor, initial_state: torch.Tensor=None, output_final_state: bool=False, cu_seqlens: torch.LongTensor | None=None) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, D = x.shape
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    o = torch.empty_like(x)
    final_state = x.new_empty(N, D) if output_final_state else None

    def grid(meta):
        return (triton.cdiv(D, meta['BD']), N)
    fused_recurrent_hgrn_fwd_kernel[grid](x=x, g=g, o=o, h0=initial_state, ht=final_state, cu_seqlens=cu_seqlens, T=T, D=D)
    return (o, final_state)

def test_fused_recurrent_hgrn_fwd_kernel():
    B, T, D = (4, 2048, 512)
    print(f'Test Configuration: B={B}, T={T}, D={D}')
    print(f'Input shape: ({B}, {T}, {D})')
    device = torch.device('npu:0')
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_vv_ops_2/fused_recurrent_hgrn_fwd_kernel'
    inputs_dict = load_fixture(os.path.join(load_dir, 'inputs.pt'))
    x = inputs_dict['x'].to(device)
    g = inputs_dict['g'].to(device)
    h0 = inputs_dict['h0'].to(device)
    output_final_state = inputs_dict['output_final_state']
    print(f'\nInputs loaded from {load_dir}/inputs.pt')
    print(f'x shape: {x.shape}, dtype: {x.dtype}')
    print(f'g shape: {g.shape}, dtype: {g.dtype}')
    print(f'h0 shape: {h0.shape}, dtype: {h0.dtype}')
    o, ht = fused_recurrent_hgrn_fwd(x=x, g=g, initial_state=h0, output_final_state=output_final_state)
    torch.npu.synchronize()
    expected_outputs = load_fixture(os.path.join(load_dir, 'outputs.pt'))
    expected_o = expected_outputs['o'].to(device)
    expected_ht = expected_outputs['ht'].to(device) if expected_outputs['ht'] is not None else None
    print('\nComparing outputs...')
    validate_cmp('float32', o, expected_o)
    torch.testing.assert_close(o, expected_o, atol=0.008, rtol=0.008)
    if expected_ht is not None:
        validate_cmp('float32', ht, expected_ht)
        torch.testing.assert_close(ht, expected_ht, atol=0.008, rtol=0.008)
    print('All outputs match within tolerance!')
    print('\n' + '=' * 80)
    print('Starting Profiling Test')
    print('=' * 80)
    experimental_config = torch_npu.profiler._ExperimentalConfig(aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization, profiler_level=torch_npu.profiler.ProfilerLevel.Level1, l2_cache=False)
    result_dir = './result_dir_fused_recurrent_hgrn_fwd_kernel'
    with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.NPU], with_stack=False, record_shapes=False, profile_memory=False, schedule=torch_npu.profiler.schedule(wait=1, warmup=1, active=30, repeat=1, skip_first=1), experimental_config=experimental_config, on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(result_dir)) as prof:
        prof.start()
        for i in range(35):
            o, ht = fused_recurrent_hgrn_fwd(x=x, g=g, initial_state=h0, output_final_state=True)
            torch.npu.synchronize()
            prof.step()
        prof.stop()
    print(f'\nProfiling completed!')
    print(f'Results saved to: {result_dir}')
    print(f'\nOutput shape: {o.shape}')
    print(f"Final state shape: {(ht.shape if ht is not None else 'None')}")
    print('\nTest completed successfully!')
