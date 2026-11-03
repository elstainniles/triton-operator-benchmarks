# Test case for chunk_hgrn_fwd_kernel_o
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

@triton.jit(do_not_specialize=['T'])
def chunk_hgrn_fwd_kernel_o(gc, o, s_b, s_t, s_d, T, D: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr):
    i_d, i_b = (tl.program_id(0), tl.program_id(1))
    o_d = i_d * BD + tl.arange(0, BD)
    mask = o_d < D
    for i_t in range(1, tl.cdiv(T, BT)):
        p_gc = tl.make_block_ptr(gc + i_b * s_b, (T, D), (s_t, s_d), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
        p_o = tl.make_block_ptr(o + i_b * s_b, (T, D), (s_t, s_d), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
        b_h0 = tl.load(o + i_b * T * D + i_t * BT * D - D + o_d, mask=mask, other=0).to(tl.float32)
        b_gc = tl.load(p_gc, boundary_check=(0, 1)).to(tl.float32)
        b_o = tl.load(p_o, boundary_check=(0, 1)).to(tl.float32)
        b_o = b_o + exp(b_gc) * b_h0[None, :]
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

def chunk_hgrn_fwd_o_call(gc: torch.Tensor, o: torch.Tensor):
    B, T, D = o.shape
    BT, BD = (128, min(64, triton.next_power_of_2(D)))
    num_warps = 8 if BD == 64 else 4

    def grid(meta):
        return (triton.cdiv(D, meta['BD']), B)
    chunk_hgrn_fwd_kernel_o[grid](gc, o, o.stride(-3), o.stride(-2), o.stride(-1), T=T, D=D, BT=BT, BD=BD, num_warps=num_warps)
    return o

def test_chunk_hgrn_fwd_h():
    B, T, D = (4, 2048, 512)
    print(f'Test Configuration: B={B}, T={T}, D={D}')
    print(f'Input shape: ({B}, {T}, {D})')
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_vv_ops_2/chunk_hgrn_fwd_kernel_o'
    device = torch.device('npu:0')
    x = load_fixture(os.path.join(load_dir, 'x.pt')).to(device)
    g = load_fixture(os.path.join(load_dir, 'g.pt')).to(device)
    h0 = load_fixture(os.path.join(load_dir, 'h0.pt')).to(device)
    o = load_fixture(os.path.join(load_dir, 'o.pt')).to(device)
    gc = load_fixture(os.path.join(load_dir, 'gc.pt')).to(device)
    print(f'\nInput tensors loaded from {load_dir}')
    print(f'x shape: {x.shape}, dtype: {x.dtype}')
    print(f'g shape: {g.shape}, dtype: {g.dtype}')
    print(f'h0 shape: {h0.shape}, dtype: {h0.dtype}')
    print(f'o shape: {o.shape}, dtype: {o.dtype}')
    print(f'gc shape: {gc.shape}, dtype: {gc.dtype}')
    o_out = chunk_hgrn_fwd_o_call(gc=gc, o=o.clone())
    expected_o_out = load_fixture(os.path.join(load_dir, 'o_out.pt')).to(device)
    print('Comparing output with expected value...')
    validate_cmp('float32', o_out, expected_o_out)
    torch.testing.assert_close(o_out, expected_o_out, atol=0.008, rtol=0.008)
    print('✓ Output verification passed!')
    print('\n' + '=' * 80)
    print('Starting Profiling Test for chunk_hgrn_fwd_kernel_o')
    print('=' * 80)
    experimental_config = torch_npu.profiler._ExperimentalConfig(aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization, profiler_level=torch_npu.profiler.ProfilerLevel.Level1, l2_cache=False)
    result_dir = './result_dir_chunk_hgrn_fwd_kernel_o'
    with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.NPU], with_stack=False, record_shapes=False, profile_memory=False, schedule=torch_npu.profiler.schedule(wait=1, warmup=1, active=30, repeat=1, skip_first=1), experimental_config=experimental_config, on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(result_dir)) as prof:
        prof.start()
        for i in range(35):
            o_out = chunk_hgrn_fwd_o_call(gc=gc, o=o.clone())
            torch.npu.synchronize()
            prof.step()
        prof.stop()
    print(f'\nProfiling completed!')
    print(f'Results saved to: {result_dir}')
    print(f'\nOutput shape: {o_out.shape}')
    print('\nTest completed successfully!')
