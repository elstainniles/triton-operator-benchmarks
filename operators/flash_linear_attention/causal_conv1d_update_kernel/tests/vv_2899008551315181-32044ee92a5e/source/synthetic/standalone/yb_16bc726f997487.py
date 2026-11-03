
from opbench.fla_fixtures import load_fixture
import os
import pytest
import torch
import torch_npu
from typing import Optional
from yb_ae727972f7e692 import causal_conv1d_update

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
        torch.testing.assert_close(y_ref.to(torch.float32), y_cal.to(torch.float32), rtol=0.01, atol=0.001, equal_nan=True)
    elif dtype == 'float32':
        torch.testing.assert_close(y_ref, y_cal, rtol=1e-05, atol=1e-05, equal_nan=True)
    elif dtype == 'int32' or dtype == 'int64' or dtype == 'int16' or (dtype == 'int8'):
        assert torch.equal(y_cal, y_ref)
    elif dtype == 'bool':
        assert torch.equal(y_cal, y_ref)
    else:
        raise ValueError('Invalid parameter "dtype" is found : {}'.format(dtype))

def compare_data_precision(dict_ref: dict, dict_cal: dict, device_type: str):
    keys_ref, keys_cal = (set(dict_ref.keys()), set(dict_cal.keys()))
    if not keys_ref.issubset(keys_cal):
        raise ValueError('The keys of dict_ref is not subset of dict_cal')
    for key in dict_ref.keys():
        val_a, val_b = (dict_ref[key], dict_cal[key])
        if type(val_a) != type(val_b):
            raise ValueError('The data type of two dicts are different')
        if isinstance(val_a, torch.Tensor):
            validate_cmp(dtype=str(val_a.dtype).split('.')[-1], y_ref=val_a, y_cal=val_b, device_type=device_type)
        else:
            raise ValueError('Non-tensor type is not currently supported')

def convert_tensor_with_device_type(indata: dict, device_type: str):
    target_device = torch.device(device_type)
    outdata = {}
    for key, value in indata.items():
        if isinstance(value, torch.Tensor):
            if value.device.type != target_device.type:
                outdata[key] = value.to(target_device)
            else:
                outdata[key] = value
        else:
            outdata[key] = value
    return outdata

@pytest.mark.functiontest
def test_causal_conv1d_update_kernel(load_path):
    data = load_fixture(os.path.join(load_path, 'test_causal_conv1d_update_kernel_cuda.pt'), map_location=torch.device('cpu'), weights_only=False)
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    y, cache_out = causal_conv1d_update(input_data['x'], input_data['cache'], weight=input_data['weight'], bias=input_data['bias'])
    npu_out = {}
    npu_out['y'] = y
    npu_out['cache'] = cache_out
    compare_data_precision(data['gpu_output'], npu_out, device_type='cpu')
    print('Test causal_conv1d_update_kernel passed!')
