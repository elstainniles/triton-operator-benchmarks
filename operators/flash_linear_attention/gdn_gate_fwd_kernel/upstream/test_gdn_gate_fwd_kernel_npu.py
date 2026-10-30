import os
import torch
import torch_npu
import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice
from typing import Optional

os.environ['TRITON_F32_DEFAULT'] = 'ieee'


if os.environ.get('FLA_USE_FAST_OPS', '0') == '1':
    @triton.jit
    def exp(x): return tldevice.fast_expf(x.to(tl.float32))
else:
    @triton.jit
    def exp(x): return tl.exp(x.to(tl.float32))


@triton.jit
def softplus(x):
    return tl.where(x < 20.0, tl.math.log(1 + tl.math.exp(x)), x)


@triton.heuristics({
    'HAS_BIAS': lambda args: args['dt_bias'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BT': BT}, num_warps=num_warps, num_stages=num_stages)
        for BT in [32, 64, 128]
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3]
    ],
    key=['H'],
)
@triton.jit(do_not_specialize=['T'])
def gdn_gate_fwd_kernel(
    g,
    A_log,
    dt_bias,
    yg,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    i_t, i_h = tl.program_id(0), tl.program_id(1)

    b_A = tl.load(A_log + i_h).to(tl.float32)

    p_g = tl.make_block_ptr(g + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_yg = tl.make_block_ptr(yg + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
    if HAS_BIAS:
        b_g = b_g + tl.load(dt_bias + i_h).to(tl.float32)
    b_yg = -exp(b_A) * softplus(b_g)
    tl.store(p_yg, b_yg.to(p_yg.dtype.element_ty), boundary_check=(0,))


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


def test_gdn_gate_fwd_kernel():
    ptfile_path = "/data/PytorchFile/Inference_or_Training_solution/fla_vv_ops_2/gdn_gate_fwd_kernel/test_gdn_gate_fwd_kernel_cuda.pt"
    data = torch.load(ptfile_path, map_location=torch.device('cpu'), weights_only=False)

    device = torch.device('npu')
    input_data = convert_tensor_with_device_type(data["input_data"], device_type='npu')

    B = input_data['B']
    T = input_data['T']
    H = input_data['H']

    yg = torch.empty(B * T, H, dtype=torch.float32, device=device)

    def grid(meta):
        return (triton.cdiv(B * T, meta['BT']), H)

    gdn_gate_fwd_kernel[grid](
        input_data['g'],
        input_data['A_log'],
        input_data['dt_bias'],
        yg,
        B * T,
        H,
    )

    npu_out = {}
    npu_out['yg'] = yg

    compare_data_precision(data["gpu_output"], npu_out, device_type='cpu')
    print("Test gdn_gate_fwd_kernel passed!")


if __name__ == "__main__":
    test_gdn_gate_fwd_kernel()
