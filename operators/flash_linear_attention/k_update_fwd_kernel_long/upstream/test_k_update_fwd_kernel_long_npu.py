import torch
import torch_npu
import triton
import triton.language as tl
from typing import Optional
import os
os.environ['TRITON_F32_DEFAULT'] = 'ieee'

NUM_WARPS_AUTOTUNE = [2, 4, 8, 16, 32]


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=w, num_stages=s)
        for w in NUM_WARPS_AUTOTUNE
        for s in [1, 2, 3]
    ],
    key=['BD', 'BT'],
)
@triton.jit
def k_update_fwd_kernel_long(
    k, a, ka, out,
    cu_seqlens, chunk_indices,
    T, D,
    BD: tl.constexpr, BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_d, i_t_blk, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    if IS_VARLEN:
        i_n, i_t_blk = tl.load(chunk_indices + i_t_blk * 2).to(tl.int32), \
            tl.load(chunk_indices + i_t_blk * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        t_start = i_t_blk * BT
        t_end = tl.minimum(t_start + BT, eos - bos)
    else:
        bos = i_b * T
        eos = (i_b + 1) * T
        t_start = i_t_blk * BT
        t_end = tl.minimum(t_start + BT, T)

    o_d = i_d * BD + tl.arange(0, BD)
    m_d = o_d < D

    for t in range(t_start, t_end):
        global_t = bos + t
        off = global_t * D + o_d
        b_k = tl.load(k + off, mask=m_d, other=0.).to(tl.float32)
        b_a = tl.load(a + off, mask=m_d, other=0.).to(tl.float32)
        b_ka = tl.load(ka + o_d, mask=m_d, eviction_policy='evict_last').to(tl.float32)
        out_val = b_k * (1 + (b_a - 1) * b_ka)
        tl.store(out + off, out_val.to(out.dtype.element_ty), mask=m_d)


def validate_cmp(dtype, y_cal, y_ref, overflow_mode: Optional[str] = None, device_type: Optional[str] = None):
    if device_type is not None:
        target_device = torch.device(device_type)
        y_cal = y_cal.to(target_device)
        y_ref = y_ref.to(target_device)
    else:
        y_cal = y_cal.npu()
        y_ref = y_ref.npu()
    if overflow_mode == "saturate":
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
        torch.testing.assert_close(y_ref, y_cal, rtol=1e-03, atol=1e-03, equal_nan=True)
    elif dtype == 'bfloat16':
        torch.testing.assert_close(y_ref.to(torch.float32), y_cal.to(torch.float32), rtol=5e-03, atol=5e-03,
                                   equal_nan=True)
    elif dtype == 'float32':
        torch.testing.assert_close(y_ref, y_cal, rtol=1e-5, atol=1e-5, equal_nan=True)
    elif dtype == 'int32' or dtype == 'int64' or dtype == 'int16' or dtype == 'int8':
        assert torch.equal(y_cal, y_ref)
    elif dtype == 'bool':
        assert torch.equal(y_cal, y_ref)
    else:
        raise ValueError('Invalid parameter "dtype" is found : {}'.format(dtype))


def compare_data_precision(dict_ref: dict, dict_cal: dict, device_type: str):
    keys_ref, keys_cal = set(dict_ref.keys()), set(dict_cal.keys())
    if not keys_ref.issubset(keys_cal):
        raise ValueError("The keys of dict_ref is not subset of dict_cal")

    for key in dict_ref.keys():
        val_a, val_b = dict_ref[key], dict_cal[key]
        if val_a is None and val_b is None:
            continue
        if type(val_a) != type(val_b):
            raise ValueError("The data type of two dicts are different")

        if isinstance(val_a, torch.Tensor):
            validate_cmp(dtype=str(val_a.dtype).split('.')[-1], y_ref=val_a, y_cal=val_b, device_type=device_type)
        else:
            raise ValueError("Non-tensor type is not currently supported")


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


def test_k_update_fwd_kernel_long():
    ptfile_path = "/data/PytorchFile/Inference_or_Training_solution/fla_vv_ops_2/k_update_fwd_kernel_long/test_k_update_fwd_kernel_long_cuda.pt"
    data = torch.load(ptfile_path, map_location=torch.device('cpu'), weights_only=False)

    device = torch.device('npu')
    input_data = convert_tensor_with_device_type(data["input_data"], device_type='npu')

    B = input_data['B']
    T = input_data['T']
    D = input_data['D']
    BD = input_data['BD']
    BT = input_data['BT']
    NT = input_data['NT']

    out = torch.empty(B, T, 64*D, dtype=torch.float32, device=device)

    N = B

    def grid(meta):
        return (triton.cdiv(D, meta['BD']), NT, N)

    k_update_fwd_kernel_long[grid](
        input_data['k'], input_data['a'], input_data['ka'], out,
        None, None,
        T, D,
        BD=BD, BT=BT,
    )

    npu_out = {}
    npu_out['out'] = out

    compare_data_precision(data["gpu_output"], npu_out, device_type='cpu')
    print("Test k_update_fwd_kernel_long passed!")


if __name__ == "__main__":
    test_k_update_fwd_kernel_long()
