
from opbench.fla_fixtures import load_fixture
import torch
import torch_npu
import triton
import triton.language as tl
from typing import Optional
import os
os.environ['TRITON_F32_DEFAULT'] = 'ieee'
NUM_WARPS_AUTOTUNE = [2, 4, 8, 16, 32]
from ofla_08a10575df4b02 import k_update_bwd_kernel_long

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

def compare_data_precision(dict_ref: dict, dict_cal: dict, device_type: str):
    keys_ref, keys_cal = (set(dict_ref.keys()), set(dict_cal.keys()))
    if not keys_ref.issubset(keys_cal):
        raise ValueError('The keys of dict_ref is not subset of dict_cal')
    for key in dict_ref.keys():
        val_a, val_b = (dict_ref[key], dict_cal[key])
        if val_a is None and val_b is None:
            continue
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

def test_k_update_bwd_kernel_long():
    ptfile_path = '/data/PytorchFile/Inference_or_Training_solution/fla_vv_ops_2/k_update_bwd_kernel_long/test_k_update_bwd_kernel_long_cuda.pt'
    data = load_fixture(ptfile_path, map_location=torch.device('cpu'), weights_only=False)
    device = torch.device('npu')
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    B = input_data['B']
    T = input_data['T']
    D = input_data['D']
    BD = input_data['BD']
    BT = input_data['BT']
    NT = input_data['NT']
    dk = input_data['dk']
    da = input_data['da']
    dka_tmp = input_data['dka_tmp']
    grad_out = input_data['grad_out']
    k = input_data['k']
    a = input_data['a']
    ka = input_data['ka']
    N = B

    def grid(meta):
        return (triton.cdiv(D, meta['BD']), NT, N)
    k_update_bwd_kernel_long[grid](grad_out, k, a, ka, dk, da, dka_tmp, None, None, T, D, BD=BD, BT=BT)
    dka = dka_tmp.sum(dim=(0, 1)).type_as(input_data['ka'])
    npu_out = {}
    npu_out['dk'] = dk
    npu_out['da'] = da
    npu_out['dka'] = dka
    compare_data_precision(data['gpu_output'], npu_out, device_type='cpu')
    print('Test k_update_bwd_kernel_long passed!')
