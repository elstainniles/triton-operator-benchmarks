
from opbench.fla_fixtures import load_fixture
import os
import pytest
import torch
import torch_npu
from typing import Optional
from yb_8696e5ac02b2ad import token_shift_fwd
from yb_8696e5ac02b2ad import token_shift_bwd
NPU_ID = os.environ.get('BENCH_NPU_ID', '0')
torch.npu.set_device(int(NPU_ID))
DEVICE = f'npu:{torch.npu.current_device()}'
D = 4096
CONFIGS = [{'B': 4, 'T': 8192, 'kernel_type': 'long'}]

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

@pytest.mark.functiontest
@pytest.mark.parametrize('cfg', CONFIGS, ids=lambda c: f"B{c['B']}_T{c['T']}_{c['kernel_type']}")
def test_token_shift_bwd_functiontest(load_dir, cfg):
    B, T, ktype = (cfg['B'], cfg['T'], cfg['kernel_type'])
    inputs = load_fixture(os.path.join(load_dir, f'token_shift_bwd_kernel_{ktype}_in.pt'))
    x = inputs['x'].to(DEVICE)
    dy = inputs['dy'].to(DEVICE)
    fwd_outputs = load_fixture(os.path.join(load_dir, f'token_shift_fwd_kernel_{ktype}_out.pt'))
    N = fwd_outputs['N']
    T_actual = fwd_outputs['T_actual']
    use_short_kernel = fwd_outputs['use_short_kernel']
    dx, grad_cache = token_shift_bwd(dy, N, T_actual, use_short_kernel=use_short_kernel)
    expected_outputs = load_fixture(os.path.join(load_dir, f'token_shift_bwd_kernel_{ktype}_out.pt'))
    expected_dx = expected_outputs['dx'].to(DEVICE)
    expected_grad_cache = expected_outputs['grad_cache'].to(DEVICE) if expected_outputs['grad_cache'] is not None else None
    validate_cmp('float16', dx, expected_dx)
    if grad_cache is not None and expected_grad_cache is not None:
        validate_cmp('float16', grad_cache, expected_grad_cache)
    elif grad_cache is not None or expected_grad_cache is not None:
        raise AssertionError(f'grad_cache mismatch: {grad_cache} vs {expected_grad_cache}')
    print(f'token_shift_bwd (B={B}, T={T}, D={D}, kernel_type={ktype}) accuracy check passed.')
