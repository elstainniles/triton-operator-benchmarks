# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch_npu
import triton
import triton.language as tl
import torch.nn.functional as F
import os
from typing import Optional

RCP_LN2 = 1.4426950216

IS_NVIDIA_HOPPER = False
BKV_LIST = [32]
NUM_WARPS = [2, 4, 8]

autotune_cache_kwargs = {}


@triton.jit
def exp(x): return tl.exp(x.to(tl.float32))


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_G_GAMMA': lambda args: args['g_gamma'] is not None,
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BV in BKV_LIST
        for num_warps in NUM_WARPS
        for num_stages in [1]
    ],
    key=['H', 'K', 'V', 'BT'],
)
@triton.jit(do_not_specialize=['T'])
def fused_chunk_fwd_kernel(
    q,
    k,
    v,
    g,
    g_gamma,
    o,
    h0,
    ht,
    cu_seqlens,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_k, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_h = i_nh // H, i_nh % H

    all = B * T
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
    NT = tl.cdiv(T, BT)

    o_i = tl.arange(0, BT)

    if USE_G_GAMMA:
        # decay rate given the head index
        b_gamma = tl.load(g_gamma + i_h)
        b_g = b_gamma * (o_i + 1)
        b_g_last = b_gamma * BT
        b_gq = exp(b_g)
        b_gk = exp(b_g_last - b_g)
        b_gn = exp(b_g_last)

    # [BT, BT]
    m_s = o_i[:, None] >= o_i[None, :]

    q = q + (bos*H + i_h) * K
    k = k + (bos*H + i_h) * K
    v = v + (bos*H + i_h) * V
    o = o + (i_k * all + bos).to(tl.int64) * H*V + i_h * V

    # [BK, BV]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h = tl.make_block_ptr(h0 + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h = tl.load(p_h, boundary_check=(0, 1)).to(tl.float32)

    for i_t in range(0, NT):
        p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k, (K, T), (1, H*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_v = tl.make_block_ptr(v, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_o = tl.make_block_ptr(o, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        last_idx = min(i_t * BT + BT, T) - 1

        # [BT, BT]
        b_s = tl.dot(b_q, b_k)

        # scalar decay
        if USE_G:
            p_g = g + (bos + o_t) * H + i_h
            b_g = tl.load(p_g, mask=(o_t < T), other=0.)
            b_g_last = tl.load(g + (bos + last_idx) * H + i_h)

            b_gq = exp(b_g)
            b_gk = exp(b_g_last - b_g)
            b_gn = exp(b_g_last)
        if USE_G_GAMMA:
            b_g_last = b_gamma * min(BT, T - i_t * BT)
            b_gk = exp(b_g_last - b_g)
            b_gn = exp(b_g_last)
        if USE_G or USE_G_GAMMA:
            b_gs = tl.where(m_s & m_t, exp(b_g[:, None] - b_g[None, :]), 0)
            # [BT, BT]
            b_s *= b_gs
            # [BT, BV]
            b_o = tl.dot(b_s.to(b_q.dtype), b_v) + tl.dot(b_q, b_h.to(b_q.dtype)) * b_gq[:, None]
            b_v = (b_v * b_gk[:, None]).to(b_v.dtype)
            b_h *= b_gn
        else:
            # [BT, BT]
            b_s *= m_s & m_t
            # [BT, BV]
            b_o = tl.dot(b_s.to(b_q.dtype), b_v) + tl.dot(b_q, b_h.to(b_q.dtype))

        b_h += tl.dot(b_k, b_v)

        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


def fused_chunk_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    g_gamma: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
):
    B, T, H, K, V = *q.shape, v.shape[-1]
    BT = chunk_size
    BK = min(max(triton.next_power_of_2(K), 16), 32)
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    NK = triton.cdiv(K, BK)

    o = v.new_empty(NK, *v.shape, dtype=torch.float) if NK > 1 else torch.empty_like(v)
    ht = k.new_empty(N, H, K, V, dtype=torch.float) if output_final_state else None
    def grid(meta): return (triton.cdiv(V, meta['BV']), NK, N * H)
    fused_chunk_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        g_gamma=g_gamma,
        o=o,
        h0=initial_state,
        ht=ht,
        cu_seqlens=cu_seqlens,
        scale=scale,
        B=B,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
    )
    if NK > 1:
        o = o.sum(0).to(v)
    return o, ht


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


def cal_precision(input: torch.tensor, golden: torch.tensor):
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


def test_fused_chunk_fwd_kernel():
    ptfile_path = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/fused_chunk_fwd_kernel_fp64/test_fused_chunk_fwd_kernel_v2.pt'
    ptfile_path1 = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/fused_chunk_fwd_kernel_fp64/test_fused_chunk_fwd_kernel_fp64_v2.pt'
    data = torch.load(ptfile_path, map_location=torch.device('cpu'), weights_only=False)
    datafp64 = torch.load(ptfile_path1, map_location=torch.device('cpu'), weights_only=False)
    
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    
    k = input_data['k']
    scale = input_data['scale']
    q = input_data['q']
    output_final_state = input_data['output_final_state']
    v = input_data['v']
    g = input_data['g']
    initial_state = input_data['initial_state']
    o, ht = fused_chunk_fwd(q=q, k=k, v=v, g=g, scale=scale, initial_state=initial_state, output_final_state=True, )
    npu_out = {
        'o': o.cpu(),
        'ht': ht.cpu() if ht is not None else None,
    }
    
    # Compare with GPU fp64 output
    for key in datafp64['gpu_output']:
        if key in npu_out:
            gpu_output = data['gpu_output'][key]
            gpu_output_fp64 = datafp64['gpu_output'][key]
            npu_output = npu_out[key]
            print(f"Comparing {key}:")
            # cal_precision(npu_output, gpu_output_fp64)
            relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(npu_output, gpu_output_fp64)
            relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(gpu_output, gpu_output_fp64)
            print(
                f"最大相对误差比 :{relative_max_npu / relative_max_gpu}, 平均相对误差比 :{relative_mean_npu / relative_mean_gpu}, 均方根误差比 :{rmse_npu / rmse_gpu}")
            assert relative_max_npu / relative_max_gpu <= 2, "最大相对误差比 check failed"
            assert relative_mean_npu / relative_mean_gpu <= 1.2, "平均相对误差比 check failed"
            assert rmse_npu / rmse_gpu <= 1.2, "均方根误差比 check failed"
    print("Output comparison passed!")



if __name__ == '__main__':
    test_fused_chunk_fwd_kernel()
