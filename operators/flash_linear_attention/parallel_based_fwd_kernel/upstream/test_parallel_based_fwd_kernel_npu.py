import os
import torch
import torch_npu
import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice
from typing import Optional
from enum import Enum
from typing import TYPE_CHECKING
import inspect
import functools
from packaging import version

if TYPE_CHECKING:
    from fla import __version__

FLA_CACHE_RESULTS = os.getenv('FLA_CACHE_RESULTS', '1') == '1'
SUPPORTS_AUTOTUNE_CACHE = "cache_results" in inspect.signature(triton.autotune).parameters
autotune_cache_kwargs = {"cache_results": FLA_CACHE_RESULTS} if SUPPORTS_AUTOTUNE_CACHE else {}

device_platform = 'cuda'
IS_AMD = (device_platform == 'hip')
IS_NVIDIA = (device_platform == 'cuda')
USE_CUDA_GRAPH = (IS_NVIDIA and os.environ.get('FLA_USE_CUDA_GRAPH', '0') == '1')


@triton.jit(do_not_specialize=['T'])
def parallel_based_fwd_kernel(
    q,
    k,
    v,
    o,
    z,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BTL: tl.constexpr,
    BTS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_kv, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    NV = tl.cdiv(V, BV)
    i_k = i_kv // (NV)
    i_v = i_kv % (NV)

    p_q = tl.make_block_ptr(q + i_bh * T*K, (T, K), (K, 1), (i_c * BTL, i_k * BK), (BTL, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * T*K, (K, T), (1, K), (i_k * BK, 0), (BK, BTS), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * T*V, (T, V), (V, 1), (0, i_v * BV), (BTS, BV), (1, 0))

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)
    b_o = tl.zeros([BTL, BV], dtype=tl.float32)
    b_z = tl.zeros([BTL], dtype=tl.float32)

    for _ in range(0, i_c * BTL, BTS):
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_s = tl.dot(b_q, (b_k), allow_tf32=False)
        b_s = 1 + b_s + 0.5 * b_s * b_s
        b_z += tl.sum(b_s, axis=1)
        b_o = b_o + tl.dot(b_s.to(b_v.dtype), b_v, allow_tf32=False)
        p_k = tl.advance(p_k, (0, BTS))
        p_v = tl.advance(p_v, (BTS, 0))

    tl.debug_barrier()
    o_q = tl.arange(0, BTL)

    o_k = tl.arange(0, BTS)
    p_k = tl.make_block_ptr(k + i_bh * T*K, (K, T), (1, K), (i_k * BK, i_c * BTL), (BK, BTS), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * T*V, (T, V), (V, 1), (i_c * BTL, i_v * BV), (BTS, BV), (1, 0))
    for _ in range(i_c * BTL, (i_c + 1) * BTL, BTS):
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        m_s = o_q[:, None] >= o_k[None, :]
        b_s = tl.dot(b_q, b_k, allow_tf32=False)
        b_s = 1 + b_s + 0.5 * b_s * b_s
        b_s = tl.where(m_s, b_s, 0)
        b_z += tl.sum(b_s, axis=1)
        b_o += tl.dot(b_s.to(b_q.dtype), b_v, allow_tf32=False)

        p_k = tl.advance(p_k, (0, BTS))
        p_v = tl.advance(p_v, (BTS, 0))
        o_k += BTS

    p_o = tl.make_block_ptr(o + (i_bh + B * H * i_k) * T*V, (T, V), (V, 1), (i_c*BTL, i_v*BV), (BTL, BV), (1, 0))
    p_z = z + (i_bh + B * H * i_k) * T + i_c * BTL + tl.arange(0, BTL)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_z, b_z.to(p_z.dtype.element_ty), mask=((i_c * BTL + tl.arange(0, BTL)) < T))


def parallel_based_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
):
    BTL, BTS = 128, 32
    BK = min(128, max(triton.next_power_of_2(k.shape[-1]), 16))
    BV = min(128, max(triton.next_power_of_2(v.shape[-1]), 16))
    B, H, T, K, V = *k.shape, v.shape[-1]
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    grid = (NK * NV, triton.cdiv(T, BTL), B * H)

    o = torch.empty(NK, B, H, T, V, device=q.device)
    z = torch.empty(NK, B, H, T, device=q.device)
    parallel_based_fwd_kernel[grid](
        q, k, v, o, z,
        scale,
        B=B,
        H=H,
        T=T,
        K=K,
        V=V,
        BTL=BTL,
        BTS=BTS,
        BK=BK,
        BV=BV,
        num_warps=4,
        num_stages=2,
    )
    return o.sum(0), z.sum(0)


def cal_precision(input: torch.Tensor, golden: torch.Tensor):
    relative_max = torch.max(torch.abs(input - golden) / (torch.abs(golden) + float(1e-7)))
    print(f"max relative error: {relative_max}")
    relative_mean = torch.mean(torch.abs(input - golden) / (torch.abs(golden) + float(1e-7)))
    print(f"mean relative error: {relative_mean}")
    rmse = torch.sqrt(torch.mean((input - golden) ** 2))
    print(f"rmse: {rmse}")
    mask = (torch.abs(golden) < float(2e-14)) & (torch.abs(input - golden) > float(2e-30))
    error = torch.abs(input - golden)
    total_error = torch.sum(error[mask])
    print(f"small val error: {total_error}")
    return relative_max, relative_mean, rmse


def test_parallel_based_fwd_kernel():
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/test_parallel_based_fwd_kernel'
    inputs_dict = torch.load(os.path.join(load_dir, "inputs.pt"), map_location='cpu', weights_only=False)

    device = torch.device('npu')
    q = inputs_dict['q'].to(device)
    k = inputs_dict['k'].to(device)
    v = inputs_dict['v'].to(device)
    scale = inputs_dict['scale']

    o, z = parallel_based_fwd(q, k, v, scale=scale)

    outputs_dict = torch.load(os.path.join(load_dir, "outputs.pt"), map_location='cpu', weights_only=False)
    o_fp64 = outputs_dict['o_fp64_actual'].to(device)
    z_fp64 = outputs_dict['z_fp64_actual'].to(device)
    o_gpu = outputs_dict['o'].to(device)
    z_gpu = outputs_dict['z'].to(device)

    print("\nComparing o with expected values...")
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(o, o_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(o_gpu, o_fp64)
    print(f"o - 最大相对误差比: {relative_max_npu / relative_max_gpu}, "
          f"平均相对误差比: {relative_mean_npu / relative_mean_gpu}, "
          f"均方根误差比: {rmse_npu / rmse_gpu}")
    assert relative_max_npu / relative_max_gpu <= 2, "o 最大相对误差比 check failed"
    assert relative_mean_npu / relative_mean_gpu <= 1.2, "o 平均相对误差比 check failed"
    assert rmse_npu / rmse_gpu <= 1.2, "o 均方根误差比 check failed"

    print("\nComparing z with expected values...")
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(z, z_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(z_gpu, z_fp64)
    print(f"z - 最大相对误差比: {relative_max_npu / relative_max_gpu}, "
          f"平均相对误差比: {relative_mean_npu / relative_mean_gpu}, "
          f"均方根误差比: {rmse_npu / rmse_gpu}")
    assert relative_max_npu / relative_max_gpu <= 2, "z 最大相对误差比 check failed"
    assert relative_mean_npu / relative_mean_gpu <= 1.2, "z 平均相对误差比 check failed"
    assert rmse_npu / rmse_gpu <= 1.2, "z 均方根误差比 check failed"

    print("Test parallel_based_fwd_kernel passed!")


if __name__ == "__main__":
    test_parallel_based_fwd_kernel()
