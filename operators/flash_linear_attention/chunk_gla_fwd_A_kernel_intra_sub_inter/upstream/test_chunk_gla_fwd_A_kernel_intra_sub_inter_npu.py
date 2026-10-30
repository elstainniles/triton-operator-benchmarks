# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import torch_npu
import triton
import triton.language as tl
from typing import Any
import functools
import inspect
import contextlib

# ============== Helper Functions and Decorators ==============

# exp2 function
@triton.jit
def exp2(x): 
    return tl.math.exp2(x.to(tl.float32))

# autotune_cache_kwargs
FLA_CACHE_RESULTS = os.getenv('FLA_CACHE_RESULTS', '1') == '1'
SUPPORTS_AUTOTUNE_CACHE = "cache_results" in inspect.signature(triton.autotune).parameters
autotune_cache_kwargs = {"cache_results": FLA_CACHE_RESULTS} if SUPPORTS_AUTOTUNE_CACHE else {}

# check_shared_mem
@functools.cache
def check_shared_mem(arch: str = "none", tensor_idx: int = 0) -> bool:
    try:
        device_shared_mem_list = [
            triton.runtime.driver.active.utils.get_device_properties(i)['max_shared_mem']
            for i in range(torch.npu.device_count())
        ]
        max_shared_memory = device_shared_mem_list[tensor_idx]
        # Default threshold
        thresholds = {
            "ampere": 166912,
            "none": 102400
        }
        return max_shared_memory >= thresholds.get(arch.lower(), 102400)
    except Exception:
        return False

# fla_cache_autotune decorator (simplified version)
def fla_cache_autotune(configs, key=None, prune_configs_by=None, reset_to_zero=None, restore_value=None,
                       pre_hook=None, post_hook=None, warmup=None, rep=None, use_cuda_graph=False,
                       do_bench=None, cache_results=False):
    if key is None:
        key = []
    
    def decorator(fn):
        return triton.autotune(
            configs=configs,
            key=key,
            prune_configs_by=prune_configs_by,
            reset_to_zero=reset_to_zero,
            restore_value=restore_value,
            pre_hook=pre_hook,
            post_hook=post_hook,
            warmup=warmup,
            rep=rep,
            use_cuda_graph=use_cuda_graph,
            do_bench=do_bench,
            **autotune_cache_kwargs
        )(fn)
    
    return decorator

# prepare_chunk_indices
def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
    cu_seqlens_cpu: torch.LongTensor | None = None,
) -> torch.LongTensor:
    if cu_seqlens_cpu is not None:
        lens = torch.diff(cu_seqlens_cpu)
        indices = torch.cat([torch.arange(n, device=cu_seqlens.device)
                            for n in triton.cdiv(lens, chunk_size).tolist()])
        return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)
    lens = torch.diff(cu_seqlens)
    indices = torch.cat([torch.arange(n) for n in triton.cdiv(lens, chunk_size).tolist()])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)

# ============== Kernel Definition ==============

BK_LIST = [32, 64] if check_shared_mem() else [16, 32]

@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@fla_cache_autotune(
    configs=[
        triton.Config({'BK': BK}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64]
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['BC'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_gla_fwd_A_kernel_intra_sub_inter(
    q,
    k,
    g,
    A,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    i_i, i_j = i_c // NC, i_c % NC
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if i_t * BT + i_i * BC >= T:
        return
    if i_i <= i_j:
        return

    b_A = tl.zeros([BC, BC], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K

        p_q = tl.make_block_ptr(q + (bos*H+i_h)*K, (T, K), (H*K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_g = tl.make_block_ptr(g + (bos*H+i_h)*K, (T, K), (H*K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_k = tl.make_block_ptr(k + (bos*H+i_h)*K, (K, T), (1, H*K), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
        p_gk = tl.make_block_ptr(g + (bos*H+i_h)*K, (K, T), (1, H*K), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
        p_gn = g + (bos + i_t * BT + i_i * BC) * H*K + i_h * K + o_k

        # [BK,]
        b_gn = tl.load(p_gn, mask=m_k, other=0)
        # [BC, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_qg = b_q * exp2(b_g - b_gn[None, :]) * scale
        # [BK, BC]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_gk = tl.load(p_gk, boundary_check=(0, 1))
        b_kg = b_k * exp2(b_gn[:, None] - b_gk)
        # [BC, BC] using tf32 to improve precision here.
        b_A += tl.dot(b_qg, b_kg)

    p_A = tl.make_block_ptr(A + (bos*H + i_h)*BT, (T, BT), (H*BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
    tl.store(p_A, b_A.to(A.dtype.element_ty), boundary_check=(0, 1))


# ============== Call Function ==============

def chunk_gla_fwd_intra_gk(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
):
    B, T, H, K = k.shape
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    BC = min(16, BT)
    NC = triton.cdiv(BT, BC)

    A = q.new_empty(B, T, H, BT, dtype=torch.float)
    grid = (NT, NC * NC, B * H)
    chunk_gla_fwd_A_kernel_intra_sub_inter[grid](
        q=q,
        k=k,
        g=g,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BC=BC,
        NC=NC,
    )
    return A


# ============== Precision Comparison Function ==============

def cal_precision(input: torch.tensor, golden: torch.tensor):
    # 最大相对误差
    relative_max = torch.max(torch.abs(input-golden)/(torch.abs(golden)+float(1e-7)))
    print(f"max relative error: {relative_max}")
    # 平均相对误差
    relative_mean = torch.mean(torch.abs(input-golden)/(torch.abs(golden)+float(1e-7)))
    print(f"mean relative error: {relative_mean}")
    # 均方根误差
    rmse = torch.sqrt(torch.mean((input-golden)**2))
    print(f"rmse: {rmse}")
    # 小值域绝对误差
    mask = (torch.abs(golden) < float(2e-14) ) & (torch.abs(input-golden) > float(2e-30))
    error = torch.abs(input-golden)
    total_error = torch.sum(error[mask])
    print(f"small val error: {total_error}")
    return relative_max, relative_mean, rmse


# ============== Test Case ==============

def test_chunk_gla_fwd_A_kernel_intra_sub_inter():
    # Load input tensors from pt files
    load_dir = "/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_gla_fwd_A_kernel_intra_sub_inter"
    inputs_dict = torch.load(os.path.join(load_dir, "inputs.pt"))
    
    device = torch.device('npu:0')
    # Convert loaded tensors to NPU
    q = inputs_dict['q'].to(device)
    k = inputs_dict['k'].to(device)
    g = inputs_dict['g'].to(device)
    cu_seqlens = inputs_dict['cu_seqlens']
    chunk_size = inputs_dict['chunk_size']
    scale = inputs_dict['scale']
    
    # Execute the kernel once
    print("\nExecuting kernel...")
    A_npu = chunk_gla_fwd_intra_gk(q, k, g, scale, chunk_size=chunk_size)
    
    # Load expected output tensors from pt file
    outputs_dict = torch.load(os.path.join(load_dir, "outputs.pt"))
    A_gpu_fp64 = outputs_dict['A_fp64'].to(device)
    A_gpu = outputs_dict['A'].to(device)
    
    # Compare outputs with expected values
    print("\nComparing outputs with expected values...")
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(A_npu, A_gpu_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(A_gpu, A_gpu_fp64)
    print(f"最大相对误差比 :{relative_max_npu / relative_max_gpu}, 平均相对误差比 :{relative_mean_npu / relative_mean_gpu}, 均方根误差比 :{rmse_npu / rmse_gpu}")
    assert relative_max_npu / relative_max_gpu <= 2, "最大相对误差比 check failed"
    assert relative_mean_npu / relative_mean_gpu <= 1.2, "平均相对误差比 check failed"
    assert rmse_npu / rmse_gpu <= 1.2, "均方根误差比 check failed"
    print("Output comparison passed!")
    
    # Warm-up run
    print("\n预热运行...")
    _ = chunk_gla_fwd_intra_gk(q, k, g, scale, chunk_size=chunk_size)
    torch.npu.synchronize()
    
    # Profiling
    print("\n开始Profiling测试...")
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1, 
        l2_cache=False
    )
    
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        with_stack=False,
        record_shapes=False,
        profile_memory=False,
        schedule=torch_npu.profiler.schedule(
            wait=1,
            warmup=1,
            active=30,
            repeat=1,
            skip_first=1
        ),
        experimental_config=experimental_config,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result_dir_original")
    ) as prof:
        prof.start()
        for i in range(35):
            _ = chunk_gla_fwd_intra_gk(q, k, g, scale, chunk_size=chunk_size)
            torch.npu.synchronize()
            prof.step()
        prof.stop()
    
    print("\nProfiling完成，结果保存在 ./result_dir_original")
    print("Test completed successfully!")
    print(f"Output shape: {A_npu.shape}")


if __name__ == "__main__":
    test_chunk_gla_fwd_A_kernel_intra_sub_inter()
