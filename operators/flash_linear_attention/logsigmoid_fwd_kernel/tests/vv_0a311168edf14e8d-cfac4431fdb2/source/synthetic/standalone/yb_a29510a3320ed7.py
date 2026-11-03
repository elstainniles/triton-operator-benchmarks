
from opbench.fla_fixtures import load_fixture
import os
import pytest
import torch
import torch_npu
from typing import Optional
from yb_dce32337bf0092 import logsigmoid_fwd
NPU_ID = os.environ.get('BENCH_NPU_ID', '0')
torch.npu.set_device(int(NPU_ID))
DEVICE = f'npu:{torch.npu.current_device()}'
CONFIGS = [{'B': 4, 'T': 2048, 'D': 1024, 'dtype': torch.float32, 'temperature': 1.0, 'desc': 'temperature=1.0'}, {'B': 4, 'T': 2048, 'D': 2048, 'dtype': torch.bfloat16, 'temperature': 0.5, 'desc': 'temperature=0.5'}, {'B': 4, 'T': 2048, 'D': 2048, 'dtype': torch.bfloat16, 'temperature': 2.0, 'desc': 'temperature=2.0'}]
kernel_name = 'logsigmoid_fwd'

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
def test_logsigmoid_fwd_kernel_functiontest(load_dir):
    for cfg in CONFIGS:
        B, T, D, dtype = (cfg['B'], cfg['T'], cfg['D'], cfg['dtype'])
        temperature, desc = (cfg['temperature'], cfg['desc'])
        inputs = load_fixture(os.path.join(load_dir, f'{kernel_name}_B{B}_T{T}_D{D}_temp{temperature}_in.pt'))
        x = inputs['x'].to(DEVICE)
        y = logsigmoid_fwd(x, temperature=temperature)
        expected_outputs = load_fixture(os.path.join(load_dir, f'{kernel_name}_B{B}_T{T}_D{D}_temp{temperature}_out.pt'))
        expected_y = expected_outputs['y'].to(DEVICE)
        dtype_str = 'float32' if dtype == torch.float32 else 'bfloat16'
        validate_cmp(dtype_str, y, expected_y)
        print(f'Accuracy check passed for logsigmoid_fwd (B={B}, T={T}, D={D}, temp={temperature})')
    print(f'[{kernel_name}] Precision check completed successfully')
