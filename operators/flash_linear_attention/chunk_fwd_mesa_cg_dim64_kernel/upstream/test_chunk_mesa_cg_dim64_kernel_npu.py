# coding=utf-8
import sys
import os
import pytest
import triton
import torch
import triton.language as tl
from typing import Optional
# import torch_npu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@triton.jit
def exp(x): 
    return tl.exp(x.to(tl.float32))

device = 'npu'

os.environ['TRITON_F32_DEFAULT'] = 'ieee'
@triton.jit()
def chunk_update_once(
    b_p,
    b_k,
    b_v,
    b_m,
    b_g_exp_q,
    b_h,
    b_lamb,
):
    b_o = tl.dot((tl.dot(b_p.to(b_k.dtype), tl.trans(b_k)) * b_m).to(b_v.dtype), b_v)
    b_o += tl.dot((b_p * b_g_exp_q).to(b_h.dtype), b_h)
    if b_lamb is not None:
        b_o += b_lamb[None, :] * b_p
    return b_o

@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def chunk_fwd_mesa_cg_dim64_kernel(
    q,
    q_final,
    k,
    h,
    o,
    v,
    h_kv,
    g,
    beta,
    lamb,
    cu_seqlens,
    chunk_indices,
    T,
    max_CG_iteration: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T

    q += (bos * H + i_h) * K
    q_final += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    h += (i_tg * H + i_h).to(tl.int64) * K * K
    g += bos * H + i_h
    beta += bos * H + i_h
    lamb += i_h * K

    o += (bos * H + i_h) * K
    v += (bos * H + i_h) * K
    h_kv += (i_tg * H + i_h).to(tl.int64) * K * K

    p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_h = tl.make_block_ptr(h, (K, K), (K, 1), (0, 0), (BK, BK), (1, 0))

    b_h = tl.load(p_h, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32)

    p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
    p_beta = tl.make_block_ptr(beta, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_beta = tl.load(p_beta, boundary_check=(0,)).to(tl.float32)
    p_lamb = tl.make_block_ptr(lamb, (K,), (1,), (0,), (BK,), (0,))

    b_lamb = tl.load(p_lamb, boundary_check=(0,)).to(tl.float32)

    b_m = exp(b_g[:, None] - b_g[None, :]) * b_beta[None, :]
    b_m = tl.where((o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t[None, :]), b_m, 0)
    b_g_exp_q = tl.exp(b_g)[:, None]

    b_x = tl.zeros([BT, BK], dtype=tl.float32)
    b_p = tl.zeros([BT, BK], dtype=tl.float32)
    b_r = tl.zeros([BT, BK], dtype=tl.float32)

    b_x += b_q * 0.
    b_r += b_q
    b_p += b_r
    b_delta_old = tl.sum(b_r*b_r, axis=1)
    for i in range(max_CG_iteration):
        b_o = chunk_update_once(b_p, b_k, b_k, b_m, b_g_exp_q, b_h, b_lamb)
        alpha = b_delta_old / (tl.sum(b_p*b_o, axis=1) + 1e-5)
        b_x += alpha[:, None] * b_p
        b_r = b_r - alpha[:, None] * b_o
        b_delta_new = tl.sum(b_r*b_r, axis=1)
        b_p = b_r + (b_delta_new / (b_delta_old + 1e-5))[:, None] * b_p
        b_delta_old = b_delta_new

    p_q_final = tl.make_block_ptr(q_final, (T, K), (H*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    tl.store(p_q_final, b_x.to(p_q_final.dtype.element_ty), boundary_check=(0, 1))

    p_h_kv = tl.make_block_ptr(h_kv, (K, K), (K, 1), (0, 0), (BK, BK), (1, 0))
    b_h_kv = tl.load(p_h_kv, boundary_check=(0, 1))
    p_v = tl.make_block_ptr(v, (T, K), (H*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_o = chunk_update_once(b_x, b_k, b_v, b_m, b_g_exp_q, b_h_kv, None)
    p_o = tl.make_block_ptr(o, (T, K), (H*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

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
        torch.testing.assert_close(y_ref, y_cal, rtol=1e-03, atol=1e-05, equal_nan=True)
    elif dtype == 'bfloat16':
        torch.testing.assert_close(y_ref.to(torch.float32), y_cal.to(torch.float32), rtol=1e-02, atol=1e-04,
                                   equal_nan=True)
    elif dtype == 'float32':
        torch.testing.assert_close(y_ref, y_cal, rtol=1e-05, atol=1e-08, equal_nan=True)
    elif dtype == 'int32' or dtype == 'int64' or dtype == 'int16' or dtype == 'int8':
        assert torch.equal(y_cal, y_ref)
    elif dtype == 'bool':
        assert torch.equal(y_cal, y_ref)
    else:
        raise ValueError('Invalid parameter \"dtype\" is found : {}'.format(dtype))

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

def cal_precision(input: torch.tensor, golden: torch.tensor, name: str = ""):
    relative_max = torch.max(torch.abs(input-golden)/(torch.abs(golden)+float(1e-7)))
    print(f"{name} - max relative error: {relative_max}")
    relative_mean = torch.mean(torch.abs(input-golden)/(torch.abs(golden)+float(1e-7)))
    print(f"{name} - mean relative error: {relative_mean}")
    rmse = torch.sqrt(torch.mean((input-golden)**2))
    print(f"{name} - rmse: {rmse}")
    mask = (torch.abs(golden) < float(2e-14) ) & (torch.abs(input-golden) > float(2e-30))
    error = torch.abs(input-golden)
    total_error = torch.sum(error[mask])
    print(f"{name} - small val error: {total_error}")
    compare_tensor(input, golden)
    return relative_max, relative_mean, rmse

def compare_tensor(npu_tensor, golden_tensor):
    abs_diff = torch.abs(npu_tensor - golden_tensor)
    abs_golden = torch.abs(golden_tensor)
    rel_diff = abs_diff / (abs_golden + 1e-10)
    
    max_abs_error = abs_diff.max().item()
    max_rel_error = rel_diff.max().item()
    mean_abs_error = abs_diff.mean().item()
    mean_rel_error = rel_diff.mean().item()
    rmse = torch.sqrt((abs_diff ** 2).mean()).item()
    ret = torch.allclose(npu_tensor, golden_tensor, 5e-3, 5e-3)
    
    print(f"\n=====<<< allclose: {ret}:")
    # if not ret:
    #     print(f"{npu_tensor=}, {golden_tensor=}")
    print(f"  Max absolute error: {max_abs_error:.6e}")
    print(f"  Max relative error: {max_rel_error:.6e}")
    print(f"  Mean absolute error: {mean_abs_error:.6e}")
    print(f"  Mean relative error: {mean_rel_error:.6e}")
    print(f"  RMSE: {rmse:.6e}")


def test_chunk_mesa_cg_dim64_kernel_npu():
    ptfile_path = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_mesa_cg_fwd/test_chunk_mesa_cg_dim64_kernel_v2.pt'
    ptfile_path1 = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_mesa_cg_fwd/test_chunk_mesa_cg_dim64_kernel_fp64_v2.pt'
    try:
        data = torch.load(ptfile_path, map_location=torch.device('cpu'), weights_only=False)
        datafp64 = torch.load(ptfile_path1, map_location=torch.device('cpu'), weights_only=False)
    except Exception as e:
        pytest.fail(f"load file {ptfile_path} failed: {str(e)}")

    input_data = convert_tensor_with_device_type(data["input_data"], device_type=device)
    
    chunk_fwd_mesa_cg_dim64_kernel[data["grid"]](
        q=input_data['q'],
        q_final=input_data['q_final'],
        k=input_data['k'],
        h=input_data['h'],
        o=input_data['o'],
        v=input_data['v'],
        h_kv=input_data['h_kv'],
        g=input_data['g'],
        beta=input_data['beta'],
        lamb=input_data['lamb'],
        cu_seqlens=input_data['cu_seqlens'] if input_data['cu_seqlens'].numel() > 0 else None,
        chunk_indices=input_data['chunk_indices'] if input_data['chunk_indices'].numel() > 0 else None,
        T=input_data['T'],
        max_CG_iteration=input_data['max_CG_iteration'],
        H=input_data['H'],
        K=input_data['K'],
        BT=input_data['BT'],
        BK=input_data['BK'],
    )

    gpu_output_q_final_fp64 = datafp64["gpu_output"]["q_final"]
    gpu_output_o_fp64 = datafp64["gpu_output"]["o"]

    npu_output_q_final = input_data["q_final"].cpu()
    npu_output_o = input_data["o"].cpu()
    # print(f"{npu_output_q_final=}, {npu_output_o=}")
    gpu_output_q_final = data["gpu_output"]["q_final"]
    gpu_output_o = data["gpu_output"]["o"]

    print("\n=== q_final precision comparison ===")
    relative_max_npu_q, relative_mean_npu_q, rmse_npu_q = cal_precision(npu_output_q_final, gpu_output_q_final_fp64, "q_final NPU")
    relative_max_gpu_q, relative_mean_gpu_q, rmse_gpu_q = cal_precision(gpu_output_q_final, gpu_output_q_final_fp64, "q_final GPU fp32")
    print(f"q_final - 最大相对误差比: {relative_max_npu_q / relative_max_gpu_q}, 平均相对误差比: {relative_mean_npu_q / relative_mean_gpu_q}, 均方根误差比: {rmse_npu_q / rmse_gpu_q}")

    print("\n=== o precision comparison ===")
    relative_max_npu_o, relative_mean_npu_o, rmse_npu_o = cal_precision(npu_output_o, gpu_output_o_fp64, "o NPU")
    relative_max_gpu_o, relative_mean_gpu_o, rmse_gpu_o = cal_precision(gpu_output_o, gpu_output_o_fp64, "o GPU fp32")
    print(f"o - 最大相对误差比: {relative_max_npu_o / relative_max_gpu_o}, 平均相对误差比: {relative_mean_npu_o / relative_mean_gpu_o}, 均方根误差比: {rmse_npu_o / rmse_gpu_o}")

if __name__ == '__main__':
    test_chunk_mesa_cg_dim64_kernel_npu()
