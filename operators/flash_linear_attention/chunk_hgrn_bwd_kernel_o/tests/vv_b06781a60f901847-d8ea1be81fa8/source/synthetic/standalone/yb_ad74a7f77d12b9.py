# Test case for chunk_hgrn_bwd_kernel_o
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
def chunk_hgrn_bwd_kernel_o(g, gc, o, dx, dg, s_b, s_t, s_d, T, D: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr):
    i_d, i_b = (tl.program_id(0), tl.program_id(1))
    o_d = i_d * BD + tl.arange(0, BD)
    mask = o_d < D
    for i_t in range(tl.cdiv(T, BT) - 1, -1, -1):
        p_g = tl.make_block_ptr(g + i_b * s_b, (T, D), (s_t, s_d), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
        p_gc = tl.make_block_ptr(gc + i_b * s_b, (T, D), (s_t, s_d), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
        p_o = tl.make_block_ptr(o + i_b * s_b, (T, D), (s_t, s_d), (i_t * BT - 1, i_d * BD), (BT, BD), (1, 0))
        p_dx = tl.make_block_ptr(dx + i_b * s_b, (T, D), (s_t, s_d), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
        p_dg = tl.make_block_ptr(dg + i_b * s_b, (T, D), (s_t, s_d), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
        mask_t = mask & ((i_t + 1) * BT < T)
        b_ht = tl.load(dx + i_b * T * D + (i_t + 1) * BT * D + o_d, mask=mask_t, other=0).to(tl.float32)
        b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
        b_gc = tl.load(p_gc, boundary_check=(0, 1)).to(tl.float32)
        b_o = tl.load(p_o, boundary_check=(0, 1)).to(tl.float32)
        b_dx = tl.load(p_dx, boundary_check=(0, 1)).to(tl.float32)
        b_dx = b_dx + exp(b_gc) * b_ht[None, :]
        b_dg = b_o * b_dx * exp(b_g)
        tl.store(p_dx, b_dx.to(p_dx.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0, 1))

def chunk_hgrn_bwd_o_call(g: torch.Tensor, gc: torch.Tensor, o: torch.Tensor, dx: torch.Tensor):
    B, T, D = dx.shape
    BT, BD = (64, min(64, triton.next_power_of_2(D)))
    num_warps = 8 if BD == 64 else 4
    dg = torch.empty_like(g, dtype=torch.float)

    def grid(meta):
        return (triton.cdiv(D, meta['BD']), B)
    chunk_hgrn_bwd_kernel_o[grid](g, gc, o, dx, dg, o.stride(-3), o.stride(-2), o.stride(-1), T=T, D=D, BT=BT, BD=BD, num_warps=num_warps)
    return (dx, dg)

def test_chunk_hgrn_fwd_h():
    B, T, D = (2, 1024, 256)
    print(f'Test Configuration: B={B}, T={T}, D={D}')
    print(f'Input shape: ({B}, {T}, {D})')
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_vv_ops_2/chunk_hgrn_bwd_kernel_o'
    device = torch.device('npu:0')
    x = load_fixture(os.path.join(load_dir, 'x.pt')).to(device)
    g = load_fixture(os.path.join(load_dir, 'g.pt')).to(device)
    h0 = load_fixture(os.path.join(load_dir, 'h0.pt')).to(device)
    do = load_fixture(os.path.join(load_dir, 'do.pt')).to(device)
    o = load_fixture(os.path.join(load_dir, 'o.pt')).to(device)
    gc = load_fixture(os.path.join(load_dir, 'gc.pt')).to(device)
    dx = load_fixture(os.path.join(load_dir, 'dx.pt')).to(device)
    print(f'\nInput tensors loaded from {load_dir}')
    print(f'x shape: {x.shape}, dtype: {x.dtype}')
    print(f'g shape: {g.shape}, dtype: {g.dtype}')
    print(f'h0 shape: {h0.shape}, dtype: {h0.dtype}')
    print(f'do shape: {do.shape}, dtype: {do.dtype}')
    print(f'o shape: {o.shape}, dtype: {o.dtype}')
    print(f'gc shape: {gc.shape}, dtype: {gc.dtype}')
    print(f'dx shape: {dx.shape}, dtype: {dx.dtype}')
    dx_out, dg = chunk_hgrn_bwd_o_call(g=g, gc=gc, o=o, dx=dx.clone())
    expected_dx_out = load_fixture(os.path.join(load_dir, 'dx_out.pt')).to(device)
    expected_dg = load_fixture(os.path.join(load_dir, 'dg.pt')).to(device)
    print('Comparing outputs with expected values...')
    validate_cmp('float32', dx_out, expected_dx_out)
    validate_cmp('float32', dg, expected_dg)
    print('✓ Output verification passed!')
    print('\n' + '=' * 80)
    print('Starting Profiling Test for chunk_hgrn_bwd_kernel_o')
    print('=' * 80)
    experimental_config = torch_npu.profiler._ExperimentalConfig(aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization, profiler_level=torch_npu.profiler.ProfilerLevel.Level1, l2_cache=False)
    result_dir = './result_dir_chunk_hgrn_bwd_kernel_o'
    with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.NPU], with_stack=False, record_shapes=False, profile_memory=False, schedule=torch_npu.profiler.schedule(wait=1, warmup=1, active=30, repeat=1, skip_first=1), experimental_config=experimental_config, on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(result_dir)) as prof:
        prof.start()
        for i in range(35):
            dx_out, dg = chunk_hgrn_bwd_o_call(g=g, gc=gc, o=o, dx=dx.clone())
            torch.npu.synchronize()
            prof.step()
        prof.stop()
    print(f'\nProfiling completed!')
    print(f'Results saved to: {result_dir}')
    print(f'\nBackward output shapes:')
    print(f'dx_out shape: {dx_out.shape}')
    print(f'dg shape: {dg.shape}')
    print('\nTest completed successfully!')
