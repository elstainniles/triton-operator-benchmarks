# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional
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
        torch.testing.assert_close(y_ref, y_cal, rtol=1e-05, atol=1e-05, equal_nan=True)
    elif dtype == 'int32' or dtype == 'int64' or dtype == 'int16' or dtype == 'int8':
        assert torch.equal(y_cal, y_ref)
    elif dtype == 'bool':
        assert torch.equal(y_cal, y_ref)
    else:
        raise ValueError('Invalid parameter \"dtype\" is found : {}'.format(dtype))

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
        thresholds = {
            "ampere": 166912,
            "none": 102400
        }
        return max_shared_memory >= thresholds.get(arch.lower(), 102400)
    except Exception:
        return False

# fla_cache_autotune decorator
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

@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@fla_cache_autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3]
    ],
    key=['BK', 'BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_gla_fwd_A_kernel_intra_sub_intra(
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
    IS_VARLEN: tl.constexpr,
):
    i_t, i_i, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    i_j = i_i
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if i_t * BT + i_i * BC >= T:
        return

    o_i = tl.arange(0, BC)
    o_k = tl.arange(0, BK)
    o_A = (i_t * BT + i_i * BC + tl.arange(0, BC)) * H*BT + i_j * BC
    m_k = o_k < K
    m_A = (i_t * BT + i_i * BC + tl.arange(0, BC)) < T

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    g += (bos * H + i_h) * K
    A += (bos * H + i_h) * BT

    p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_t * BT + i_i * BC, 0), (BC, BK), (1, 0))
    p_g = tl.make_block_ptr(g, (T, K), (H*K, 1), (i_t * BT + i_i * BC, 0), (BC, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_g = tl.load(p_g, boundary_check=(0, 1))

    p_k = k + (i_t * BT + i_j * BC) * H*K + o_k
    p_gk = g + (i_t * BT + i_j * BC) * H*K + o_k

    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        b_k = tl.load(p_k, mask=m_k, other=0).to(tl.float32)
        b_gk = tl.load(p_gk, mask=m_k, other=0).to(tl.float32)
        b_A = tl.sum(b_q * b_k[None, :] * exp2(b_g - b_gk[None, :]), 1) * scale
        tl.store(A + o_A + j, b_A, mask=m_A)
        p_k += H*K
        p_gk += H*K

    tl.debug_barrier()
    b_A = tl.zeros([BC, BC], dtype=tl.float32)
    tl.store(A + o_A[:, None] + o_i, b_A, mask=m_A[:, None] & (o_i[:, None] < o_i))


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
    
    # For K <= 256, use chunk_gla_fwd_A_kernel_intra_sub_intra
    if K <= 256:
        BK = max(triton.next_power_of_2(K), 16)
        grid = (NT, NC, B * H)
        chunk_gla_fwd_A_kernel_intra_sub_intra[grid](
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
            BK=BK,
        )
    return A


# ============== Test Case ==============

def test_chunk_gla_fwd_A_kernel_intra_sub_intra():
    # Config with large shapes and K <= 256 for this kernel
    B, T, H, K = 2, 2048, 8, 128  # K=128 <= 256
    chunk_size = 64
    scale = K ** -0.5
    
    # Load input tensors from pt files
    load_dir = "/data/PytorchFile/Inference_or_Training_solution/fla_vv_ops_2/chunk_gla_fwd_A_kernel_intra_sub_intra"
    
    device = torch.device('npu:0')
    q = torch.load(os.path.join(load_dir, "q.pt")).to(device)
    k = torch.load(os.path.join(load_dir, "k.pt")).to(device)
    g = torch.load(os.path.join(load_dir, "g.pt")).to(device)
    
    # Execute the kernel once
    A = chunk_gla_fwd_intra_gk(q, k, g, scale, chunk_size=chunk_size)
    
    # Load expected output and compare
    expected_A = torch.load(os.path.join(load_dir, "A.pt")).to(device)
    
    print("Comparing output with expected value...")
    validate_cmp("float16", A, expected_A)
    print("✓ Output verification passed!")
    
    # Warm-up run
    _ = chunk_gla_fwd_intra_gk(q, k, g, scale, chunk_size=chunk_size)
    torch.npu.synchronize()
    
    # Profiling
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
    
    print("Test completed successfully!")
    print(f"Input shapes: B={B}, T={T}, H={H}, K={K}")
    print(f"Output shape: {A.shape}")
    print("Profiling results saved to ./result_dir_original")


if __name__ == "__main__":
    test_chunk_gla_fwd_A_kernel_intra_sub_intra()
