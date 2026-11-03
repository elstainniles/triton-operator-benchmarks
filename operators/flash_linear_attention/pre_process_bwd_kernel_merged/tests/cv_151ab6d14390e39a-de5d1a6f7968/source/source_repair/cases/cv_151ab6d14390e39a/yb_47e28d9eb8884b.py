
from opbench.fla_fixtures import load_fixture
import torch
import torch_npu
import triton
import triton.language as tl
USE_CUDA_GRAPH = False
autotune_cache_kwargs = {}

def check_shared_mem(arch='none', tensor_idx=0):
    return False

@triton.jit
def exp2(x):
    return tl.math.exp2(x.to(tl.float32))
from sre_93af2deb7da706 import pre_process_bwd_kernel_merged

def chunk_gated_delta_rule_bwd_dhu_pre_process(q: torch.Tensor, k: torch.Tensor, w: torch.Tensor, do: torch.Tensor, dv: torch.Tensor, g: torch.Tensor | None=None, gk: torch.Tensor | None=None, bg: torch.Tensor | None=None, scale: float | None=None, cu_seqlens: torch.LongTensor | None=None, chunk_size: int=64):
    B, T, H, K, V, HV = (*q.shape, do.shape[-1], do.shape[2])
    BT = chunk_size
    assert K <= 256, 'current kernel does not support head dimension being larger than 256.'
    BK = triton.next_power_of_2(K)
    if cu_seqlens is None:
        N = B
    else:
        N = len(cu_seqlens) - 1
    dhm = q.new_zeros(HV, K, V + K, dtype=torch.float32)
    BLOCK_SIZE = 32 if K <= 64 else 64
    grid = (triton.cdiv(V, BLOCK_SIZE) + triton.cdiv(K, BLOCK_SIZE), HV)
    pre_process_bwd_kernel_merged[grid](q=q, k=k if bg is None else bg, w=w, g=g, gk=gk, do=do, dhm=dhm, dv=dv, cu_seqlens=cu_seqlens, scale=scale, T=T, H=H, HV=HV, K=K, V=V, BT=BT, BK1=BK, BLOCK_SIZE=BLOCK_SIZE, USE_BG=bg is not None)
    return dhm

def validate_cmp(dtype, y_cal, y_ref, overflow_mode=None, device_type=None):
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

def compare_data_precision(dict_ref, dict_cal, device_type):
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

def convert_tensor_with_device_type(indata, device_type):
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

def test_pre_process_bwd_kernel_merged():
    ptfile_path = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/test_pre_process_bwd_kernel_merged/test_pre_process_bwd_kernel_merged_cuda.pt'
    data = load_fixture(ptfile_path, map_location=torch.device('cpu'), weights_only=False)
    device = torch.device('npu')
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    dhm = chunk_gated_delta_rule_bwd_dhu_pre_process(input_data['q'], input_data['k'], input_data['w'], input_data['do'], input_data['dv'], g=input_data['g'], gk=input_data['gk'], bg=input_data['bg'], scale=input_data['scale'], cu_seqlens=input_data['cu_seqlens'], chunk_size=input_data['chunk_size'])
    npu_out = {'dhm': dhm}
    compare_data_precision(data['gpu_output'], npu_out, device_type='cpu')
    print('Test pre_process_bwd_kernel_merged passed!')
