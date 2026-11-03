
from opbench.fla_fixtures import load_fixture
import os
import pytest
import torch
import torch_npu
from typing import Optional
from yb_df4e244f2240a0 import rotary_embedding_fwdbwd
from yb_df4e244f2240a0 import RotaryEmbedding
NPU_ID = os.environ.get('BENCH_NPU_ID', '0')
torch.npu.set_device(int(NPU_ID))
kernel_name = 'rotary_embedding_kernel'

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
def test_rotary_embedding_kernel_functiontest(load_dir):
    inputs = load_fixture(os.path.join(load_dir, f'{kernel_name}_in.pt'))
    x = inputs['x'].to('npu').requires_grad_(True)
    cos = inputs['cos'].to('npu')
    sin = inputs['sin'].to('npu')
    y = rotary_embedding_fwdbwd(x, cos, sin)
    expected_outputs = load_fixture(os.path.join(load_dir, f'{kernel_name}_out.pt'))
    expected_y = expected_outputs['y'].to('npu')
    validate_cmp('float16', y, expected_y)
    print('rotary_embedding_fwdbwd accuracy check passed.')
    print(f'[{kernel_name}] Precision check completed successfully')
