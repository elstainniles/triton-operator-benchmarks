# coding=utf-8
# -------------------------------------------------
# Project Name: PythonProject1
# File Name： test_qk_rmsnorm_bias_triton_kernel_npu
# Tool Name: PyCharm
# Author : l00567229
# Date： 2025/12/20 14:19
# -------------------------------------------------
import sys
import pytest
import triton
import torch
import triton.language as tl


@triton.jit
def exp(x):
    return tl.exp(x.to(tl.float32))


@triton.jit
def mesa_net_decoding_one_step_kernel(
        q,
        k,
        v,
        g,
        o,
        lamb,
        beta,
        prev_h_kk,
        prev_h_kv,
        curr_h_kk,
        curr_h_kv,
        B: tl.constexpr,
        H: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BK: tl.constexpr,
        BV: tl.constexpr,
        MAX_CG_STEP: tl.constexpr,
):
    i_nh = tl.program_id(0)
    i_h = i_nh % H

    o_k = tl.arange(0, BK)
    o_v = tl.arange(0, BV)

    p_q = q + i_nh * K + o_k
    p_k = k + i_nh * K + o_k
    p_v = v + i_nh * V + o_v
    p_beta = beta + i_nh
    p_g = g + i_nh
    p_lamb = lamb + i_h * K + o_k

    b_g = exp(tl.load(p_g).to(tl.float32))
    b_beta = tl.load(p_beta).to(tl.float32)

    mask_k = o_k < K
    mask_v = o_v < V
    mask_kk = mask_k[:, None] & mask_k[None, :]
    mask_kv = mask_k[:, None] & mask_v[None, :]

    b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
    b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
    b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
    b_lamb = tl.load(p_lamb, mask=mask_k, other=0).to(tl.float32)

    p_hkk_prev = prev_h_kk + i_nh * K * K + o_k[:, None] * K + o_k[None, :]
    b_h_kk = tl.load(p_hkk_prev, mask=mask_kk, other=0).to(tl.float32)

    b_h_kk = b_h_kk * b_g + (b_k * b_beta)[:, None] * b_k[None, :]

    p_hkk_curr = curr_h_kk + i_nh * K * K + o_k[:, None] * K + o_k[None, :]
    tl.store(p_hkk_curr, b_h_kk.to(p_hkk_curr.dtype.element_ty), mask=mask_kk)

    p_hkv_prev = prev_h_kv + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
    b_h_kv = tl.load(p_hkv_prev, mask=mask_kv, other=0).to(tl.float32)
    b_h_kv = b_h_kv * b_g + (b_k * b_beta)[:, None] * b_v[None, :]
    p_hkv_curr = curr_h_kv + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
    tl.store(p_hkv_curr, b_h_kv.to(p_hkv_curr.dtype.element_ty), mask=mask_kv)

    diag_mask = tl.arange(0, BK)[:, None] == tl.arange(0, BK)[None, :]
    diag_mask = diag_mask & mask_kk
    b_h_kk_diag = tl.sum(tl.where(diag_mask, b_h_kk, 0.0), axis=1)

    b_x = b_q / (b_h_kk_diag + b_lamb + 1e-5)
    b_Hx = tl.sum(b_h_kk * b_x[:, None], axis=0)
    b_r = b_q - b_Hx - b_lamb * b_x
    b_p = tl.zeros([BK], dtype=tl.float32)
    b_p += b_r
    delta_old = tl.sum(b_r * b_r)

    for i_iter in range(MAX_CG_STEP):
        b_Ap = tl.sum(b_h_kk * b_p[:, None], axis=0) + b_lamb * b_p
        pAp = tl.sum(b_p * b_Ap)
        alpha = delta_old / (pAp + 1e-5)
        b_x = b_x + alpha * b_p
        b_r = b_r - alpha * b_Ap
        delta_new = tl.sum(b_r * b_r)
        beta_cg = delta_new / (delta_old + 1e-5)
        b_p = b_r + beta_cg * b_p
        delta_old = delta_new
    b_o = tl.sum(b_h_kv * b_x[:, None], axis=0)
    p_o = o + i_nh * V + o_v
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)


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


def up_dtype_and_to_cpu(x):
    if isinstance(x, torch.Tensor):
        x = x.cpu()
        if x.dtype in [torch.bfloat16, torch.float16]:
            return x.to(torch.float32)
        if x.dtype == torch.float32:
            return x.to(torch.float64)
    return x


def torch_ref_cpu(data):
    q = up_dtype_and_to_cpu(data["input_data"]["q"])
    k = up_dtype_and_to_cpu(data["input_data"]["k"])
    v = up_dtype_and_to_cpu(data["input_data"]["v"])
    g = up_dtype_and_to_cpu(data["input_data"]["g"])
    lamb = up_dtype_and_to_cpu(data["input_data"]["lamb"])
    beta = up_dtype_and_to_cpu(data["input_data"]["beta"])
    prev_h_kk = up_dtype_and_to_cpu(data["input_data"]["prev_h_kk"])
    prev_h_kv = up_dtype_and_to_cpu(data["input_data"]["prev_h_kv"])

    B = data["input_data"]["B"]
    H = data["input_data"]["H"]
    K = data["input_data"]["K"]
    V = data["input_data"]["V"]
    MAX_CG_STEP = data["input_data"]["MAX_CG_STEP"]

    curr_h_kk_ref = torch.zeros(B, H, K, K, dtype=q.dtype, device='cpu')
    curr_h_kv_ref = torch.zeros(B, H, K, V, dtype=q.dtype, device='cpu')
    o_ref = torch.zeros(B, H, V, dtype=q.dtype, device='cpu')

    for b in range(B):
        for h in range(H):
            b_g = torch.exp(g[b, h])
            b_beta = beta[b, h]
            b_k = k[b, h]
            b_v = v[b, h]
            b_q = q[b, h]
            b_lamb = lamb[h]

            b_h_kk = prev_h_kk[b, h] * b_g + (b_k * b_beta)[:, None] * b_k[None, :]
            b_h_kv = prev_h_kv[b, h] * b_g + (b_k * b_beta)[:, None] * b_v[None, :]

            curr_h_kk_ref[b, h] = b_h_kk
            curr_h_kv_ref[b, h] = b_h_kv

            b_h_kk_diag = torch.diag(b_h_kk)
            b_x = b_q / (b_h_kk_diag + b_lamb + 1e-5)
            b_Hx = torch.sum(b_h_kk * b_x[:, None], dim=0)
            b_r = b_q - b_Hx - b_lamb * b_x
            b_p = b_r.clone()
            delta_old = torch.sum(b_r * b_r)

            for _ in range(MAX_CG_STEP):
                b_Ap = torch.sum(b_h_kk * b_p[:, None], dim=0) + b_lamb * b_p
                pAp = torch.sum(b_p * b_Ap)
                alpha = delta_old / (pAp + 1e-5)
                b_x = b_x + alpha * b_p
                b_r = b_r - alpha * b_Ap
                delta_new = torch.sum(b_r * b_r)
                beta_cg = delta_new / (delta_old + 1e-5)
                b_p = b_r + beta_cg * b_p
                delta_old = delta_new

            b_o = torch.sum(b_h_kv * b_x[:, None], dim=0)
            o_ref[b, h] = b_o

    return o_ref, curr_h_kk_ref, curr_h_kv_ref


def cal_precision(input: torch.Tensor, golden: torch.Tensor):
    relative_max = torch.max(torch.abs(input - golden) / (torch.abs(golden) + float(1e-7)))
    relative_mean = torch.mean(torch.abs(input - golden) / (torch.abs(golden) + float(1e-7)))
    rmse = torch.sqrt(torch.mean((input - golden) ** 2))
    return relative_max, relative_mean, rmse


@pytest.mark.functiontest
def test_kernel():
    ptfile_path = "/data/PytorchFile/Inference_or_Training_solution/fla_vv_ops_2/test_mesa_net_decoding_one_step_kernel/test_mesa_net_decoding_one_step_kernel.pt"
    try:
        data = torch.load(ptfile_path, map_location=torch.device('cpu'), weights_only=False)
    except Exception as e:
        pytest.fail(f"load file {ptfile_path} failed: {str(e)}")

    o_ref, curr_h_kk_ref, curr_h_kv_ref = torch_ref_cpu(data)

    input_data = convert_tensor_with_device_type(data["input_data"], device_type='npu')
    mesa_net_decoding_one_step_kernel[data["grid"]](**input_data)

    o_gpu = data["gpu_output"]["o"]
    curr_h_kk_gpu = data["gpu_output"]["curr_h_kk"]
    curr_h_kv_gpu = data["gpu_output"]["curr_h_kv"]

    o_npu = input_data["o"].cpu()
    curr_h_kk_npu = input_data["curr_h_kk"].cpu()
    curr_h_kv_npu = input_data["curr_h_kv"].cpu()

    mare_npu_hkk, mere_npu_hkk, rmse_npu_hkk = cal_precision(curr_h_kk_npu.to(curr_h_kk_ref.dtype), curr_h_kk_ref)
    mare_gpu_hkk, mere_gpu_hkk, rmse_gpu_hkk = cal_precision(curr_h_kk_gpu.to(curr_h_kk_ref.dtype), curr_h_kk_ref)

    mare_npu_hkv, mere_npu_hkv, rmse_npu_hkv = cal_precision(curr_h_kv_npu.to(curr_h_kv_ref.dtype), curr_h_kv_ref)
    mare_gpu_hkv, mere_gpu_hkv, rmse_gpu_hkv = cal_precision(curr_h_kv_gpu.to(curr_h_kv_ref.dtype), curr_h_kv_ref)

    mare_npu_o, mere_npu_o, rmse_npu_o = cal_precision(o_npu.to(o_ref.dtype), o_ref)
    mare_gpu_o, mere_gpu_o, rmse_gpu_o = cal_precision(o_gpu.to(o_ref.dtype), o_ref)

    print(f"\ncurr_h_kk - GPU vs CPU: mare={mare_gpu_hkk:.6e}, mere={mere_gpu_hkk:.6e}, rmse={rmse_gpu_hkk:.6e}")
    print(f"curr_h_kk - NPU vs CPU: mare={mare_npu_hkk:.6e}, mere={mere_npu_hkk:.6e}, rmse={rmse_npu_hkk:.6e}")


    print(f"\ncurr_h_kv - GPU vs CPU: mare={mare_gpu_hkv:.6e}, mere={mere_gpu_hkv:.6e}, rmse={rmse_gpu_hkv:.6e}")
    print(f"curr_h_kv - NPU vs CPU: mare={mare_npu_hkv:.6e}, mere={mere_npu_hkv:.6e}, rmse={rmse_npu_hkv:.6e}")

    print(f"\no - GPU vs CPU: mare={mare_gpu_o:.6e}, mere={mere_gpu_o:.6e}, rmse={rmse_gpu_o:.6e}")
    print(f"o - NPU vs CPU: mare={mare_npu_o:.6e}, mere={mere_npu_o:.6e}, rmse={rmse_npu_o:.6e}")

    mare_thd, mere_thd, rmse_thd = 2, 1.2, 1.2  # L0
    assert mare_npu_hkk / mare_gpu_hkk <= mare_thd
    assert mere_npu_hkk / mere_gpu_hkk <= mere_thd
    assert rmse_npu_hkk / rmse_gpu_hkk <= rmse_thd

    assert mare_npu_hkv / mare_gpu_hkv <= mare_thd
    assert mere_npu_hkv / mere_gpu_hkv <= mere_thd
    assert rmse_npu_hkv / rmse_gpu_hkv <= rmse_thd

    assert mare_npu_o / mare_gpu_o <= mare_thd
    assert mere_npu_o / mere_gpu_o <= mere_thd
    assert rmse_npu_o / rmse_gpu_o <= rmse_thd
