import os
import math
import torch
import torch_npu
import triton
import triton.language as tl
from einops import reduce
from typing import Optional


def ceil_div(x: int, y: int) -> int:
    return math.ceil(x / y)


def ceil_log(x: int, b: int) -> int:
    return math.ceil(math.log(x, b))


@triton.jit
def exp(x):
    return tl.exp(x.to(tl.float32))


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [4]
        for num_stages in [2, 3, 4]
    ],
    key=["H", "K", "V"],
    restore_value=["dk", "dg", "dg_last"],
)
@triton.jit(do_not_specialize=["T"])
def chunkwise_bwd_kernel_dkg(
    dh,
    k,
    v,
    g,
    dg_last,
    dk,
    dg,
    cu_seqlens,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    L: tl.constexpr,
    BT: tl.constexpr,
    NT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H

    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T

    o_i = tl.arange(0, BT)
    o_t = i_t * BT + o_i
    m_t = o_t < T

    p_dh = tl.make_block_ptr(
        dh + ((i_n * NT + i_t) * H + i_h) * K * V,
        (V, K),
        (1, V),
        (0, 0),
        (V, K),
        (0, 1),
    )
    p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_k = tl.make_block_ptr(k + bos * K, (T, K), (K, 1), (i_t * BT, 0), (BT, K), (1, 0))
    p_v = tl.make_block_ptr(
        v + (bos * H + i_h) * V,
        (T, V),
        (H * V, 1),
        (i_t * BT, 0),
        (BT, V),
        (1, 0),
    )
    p_dk = tl.make_block_ptr(
        dk + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT, 0), (BT, K), (1, 0),
    )
    p_dg = tl.make_block_ptr(dg + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))

    b_dh = tl.load(p_dh, boundary_check=(0, 1))
    b_g = tl.load(p_g, boundary_check=(0,))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    last_idx = min((i_t + 1) * BT, T) - 1
    b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
    p_dg_last = dg_last + i_n * NT * H + i_t * H + i_h
    b_dg_last = tl.load(p_dg_last)

    b_dg_last *= tl.exp(b_g_last)
    b_dk = tl.where(m_t, exp(b_g_last - b_g), 0)[:, None] * tl.dot(b_v, b_dh).to(b_v.dtype)
    b_dg = tl.load(p_dg, boundary_check=(0,))
    b_dg -= tl.sum(b_k * b_dk, axis=1)
    b_dg_last += tl.sum(b_dk * b_k)

    b_dg = tl.where(o_i < BT - 1, b_dg, b_dg + b_dg_last)

    tl.store(p_dg, b_dg, boundary_check=(0,))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))


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


def cal_precision(input: torch.Tensor, golden: torch.Tensor):
    relative_max = torch.max(torch.abs(input-golden)/(torch.abs(golden)+float(1e-7)))
    print(f"max relative error: {relative_max}")
    relative_mean = torch.mean(torch.abs(input-golden)/(torch.abs(golden)+float(1e-7)))
    print(f"mean relative error: {relative_mean}")
    rmse = torch.sqrt(torch.mean((input-golden)**2))
    print(f"rmse: {rmse}")
    mask = (torch.abs(golden) < float(2e-14) ) & (torch.abs(input-golden) > float(2e-30))
    error = torch.abs(input-golden)
    total_error = torch.sum(error[mask])
    print(f"small val error: {total_error}")
    return relative_max, relative_mean, rmse


def test_chunkwise_bwd_kernel_dkg():
    load_dir = "/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunkwise_bwd_kernel_dkg"
    inputs_dict = torch.load(os.path.join(load_dir, "inputs.pt"))
    
    device = torch.device('npu:0')
    dh = inputs_dict['dh'].to(device)
    k = inputs_dict['k'].to(device)
    v = inputs_dict['v'].to(device)
    g = inputs_dict['g'].to(device)
    dg_last = inputs_dict['dg_last'].to(device)
    dk = inputs_dict['dk'].to(device)
    dg = inputs_dict['dg'].to(device)
    cu_seqlens = inputs_dict['cu_seqlens']
    B = inputs_dict['B']
    T = inputs_dict['T']
    K = inputs_dict['K']
    H = inputs_dict['H']
    V = inputs_dict['V']
    L = inputs_dict['L']
    BT = inputs_dict['BT']
    NT = inputs_dict['NT']
    
    grid_t = (NT, B * H)
    
    print("\nExecuting kernel...")
    chunkwise_bwd_kernel_dkg[grid_t](
        dh=dh, k=k, v=v, g=g, dg_last=dg_last, dk=dk, dg=dg,
        cu_seqlens=cu_seqlens, T=T, H=H, K=K, V=V, L=L, BT=BT, NT=NT,
    )
    
    outputs_dict = torch.load(os.path.join(load_dir, "outputs.pt"))
    dk_gpu_fp64 = outputs_dict['dk_fp64'].to(device)
    dg_gpu_fp64 = outputs_dict['dg_fp64'].to(device)
    dk_gpu = outputs_dict['dk'].to(device)
    dg_gpu = outputs_dict['dg'].to(device)
    
    print("\nComparing outputs with expected values...")
    print("Comparing dk:")
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(dk, dk_gpu_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(dk_gpu, dk_gpu_fp64)
    print(f"最大相对误差比 :{relative_max_npu / relative_max_gpu}, 平均相对误差比 :{relative_mean_npu / relative_mean_gpu}, 均方根误差比 :{rmse_npu / rmse_gpu}")
    assert relative_max_npu / relative_max_gpu <= 2, "dk 最大相对误差比 check failed"
    assert relative_mean_npu / relative_mean_gpu <= 1.2, "dk 平均相对误差比 check failed"
    assert rmse_npu / rmse_gpu <= 1.2, "dk 均方根误差比 check failed"
    
    print("Comparing dg:")
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(dg, dg_gpu_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(dg_gpu, dg_gpu_fp64)
    print(f"最大相对误差比 :{relative_max_npu / relative_max_gpu}, 平均相对误差比 :{relative_mean_npu / relative_mean_gpu}, 均方根误差比 :{rmse_npu / rmse_gpu}")
    assert relative_max_npu / relative_max_gpu <= 2, "dg 最大相对误差比 check failed"
    assert relative_mean_npu / relative_mean_gpu <= 1.2, "dg 平均相对误差比 check failed"
    assert rmse_npu / rmse_gpu <= 1.2, "dg 均方根误差比 check failed"
    print("Output comparison passed!")


if __name__ == "__main__":
    test_chunkwise_bwd_kernel_dkg()
