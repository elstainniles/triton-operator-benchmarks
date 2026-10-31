# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import functools
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import torch_npu


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


# ============================================================================
# Helper functions from fla.utils
# ============================================================================

FLA_CACHE_RESULTS = os.getenv('FLA_CACHE_RESULTS', '1') == '1'
SUPPORTS_AUTOTUNE_CACHE = "cache_results" in functools.partial(triton.autotune, configs=[]).keywords if hasattr(functools.partial(triton.autotune, configs=[]), 'keywords') else False
autotune_cache_kwargs = {"cache_results": FLA_CACHE_RESULTS} if SUPPORTS_AUTOTUNE_CACHE else {}


def tensor_cache(fn):
    """A decorator that caches the most recent result of a function with tensor inputs."""
    last_args = None
    last_kwargs = None
    last_result = None

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        nonlocal last_args, last_kwargs, last_result
        
        if last_args is not None and last_kwargs is not None:
            if len(args) == len(last_args) and len(kwargs) == len(last_kwargs):
                if all(a is b for a, b in zip(args, last_args)) and \
                        all(k in last_kwargs and v is last_kwargs[k] for k, v in kwargs.items()):
                    return last_result
        
        result = fn(*args, **kwargs)
        last_args, last_kwargs, last_result = args, kwargs, result
        return result
    
    return wrapper


@functools.cache
def get_multiprocessor_count(tensor_idx: int = 0) -> int:
    try:
        return triton.runtime.driver.active.utils.get_device_properties(tensor_idx)['multiprocessor_count']
    except BaseException:
        if triton.runtime.driver.active.get_current_target().backend == 'npu':
            return triton.runtime.driver.active.utils.get_device_properties(tensor_idx)['num_vectorcore']
        else:
            return 1


def get_all_max_shared_mem():
    try:
        device_torch_lib = torch.npu if torch.npu.is_available() else torch.cuda
        return [
            triton.runtime.driver.active.utils.get_device_properties(i)['max_shared_mem']
            for i in range(device_torch_lib.device_count())
        ]
    except BaseException:
        return [-1]


class Backend:
    ADA = 101376
    AMPERE = 166912
    HOPPER = 232448
    DEFAULT = 102400

    @classmethod
    def get_shared_memory(cls, arch: str) -> int:
        try:
            return getattr(cls, arch.upper())
        except AttributeError:
            return cls.DEFAULT


@functools.cache
def check_shared_mem(arch: str = "none", tensor_idx: int = 0) -> bool:
    try:
        device_shared_mem_list = get_all_max_shared_mem()
        max_shared_memory = device_shared_mem_list[tensor_idx]
        return max_shared_memory >= Backend.get_shared_memory(arch)
    except Exception:
        return False


# ============================================================================
# Helper functions from fla.ops.utils.index
# ============================================================================

@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return torch.diff(cu_seqlens)


@tensor_cache
def prepare_chunk_offsets(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
) -> torch.LongTensor:
    return F.pad(triton.cdiv(prepare_lens(cu_seqlens), chunk_size), (1, 0), value=0).cumsum(-1)


@tensor_cache
def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
    cu_seqlens_cpu: torch.LongTensor | None = None,
) -> torch.LongTensor:
    if cu_seqlens_cpu is not None:
        indices = torch.cat([torch.arange(n, device=cu_seqlens.device)
                            for n in triton.cdiv(prepare_lens(cu_seqlens_cpu), chunk_size).tolist()])
        return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)
    indices = torch.cat([torch.arange(n) for n in triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


# ============================================================================
# Helper functions from fla.ops.utils.op
# ============================================================================

@triton.jit
def exp(x):
    return tl.exp(x.to(tl.float32))


@triton.jit
def log(x):
    return tl.log(x.to(tl.float32))


# ============================================================================
# Helper kernel from fla.ops.attn.parallel
# ============================================================================

@triton.jit
def parallel_attn_bwd_kernel_preprocess(
    o,
    do,
    delta,
    B: tl.constexpr,
    V: tl.constexpr,
):
    i_n = tl.program_id(0)
    o_d = tl.arange(0, B)
    m_d = o_d < V

    b_o = tl.load(o + i_n * V + o_d, mask=m_d, other=0)
    b_do = tl.load(do + i_n * V + o_d, mask=m_d, other=0).to(tl.float32)
    b_delta = tl.sum(b_o * b_do)

    tl.store(delta + i_n, b_delta.to(delta.dtype.element_ty))


def parallel_attn_bwd_preprocess(
    o: torch.Tensor,
    do: torch.Tensor,
):
    V = o.shape[-1]
    delta = torch.empty_like(o[..., 0], dtype=torch.float)
    parallel_attn_bwd_kernel_preprocess[(delta.numel(),)](
        o=o,
        do=do,
        delta=delta,
        B=triton.next_power_of_2(V),
        V=V,
    )
    return delta


# ============================================================================
# Kernel definition: parallel_nsa_compression_bwd_kernel_dq
# ============================================================================

@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4]
    ],
    key=['BS', 'BK', 'BV'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def parallel_nsa_compression_bwd_kernel_dq(
    q,
    k,
    v,
    lse,
    delta,
    do,
    dq,
    scale,
    cu_seqlens,
    token_indices,
    chunk_offsets,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BC: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    all = B * T
    if IS_VARLEN:
        i_n, i_t = tl.load(token_indices + i_t * 2).to(tl.int32), tl.load(token_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        boc = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_b * T, i_b * T + T
        boc = i_b * tl.cdiv(T, BS)

    q += (bos + i_t) * HQ*K
    do += (bos + i_t) * HQ*V
    lse += (bos + i_t) * HQ
    delta += (bos + i_t) * HQ
    dq += (i_v * all + bos + i_t) * HQ*K

    p_q = tl.make_block_ptr(q, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0))
    p_dq = tl.make_block_ptr(dq, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0))

    # [G, BK]
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)

    p_do = tl.make_block_ptr(do, (HQ, V), (V, 1), (i_h * G, i_v * BV), (G, BV), (1, 0))
    p_lse = lse + i_h * G + tl.arange(0, G)
    p_delta = delta + i_h * G + tl.arange(0, G)

    # the number of compression representations in total
    TC = tl.cdiv(T, BS)
    # the number of compression representations required to iterate over
    # incomplete compression blocks are not included
    NC = (i_t + 1) // BS

    # [G, BV]
    b_do = tl.load(p_do, boundary_check=(0, 1))
    # [G]
    b_lse = tl.load(p_lse)
    b_delta = tl.load(p_delta)

    # [G, BK]
    b_dq = tl.zeros([G, BK], dtype=tl.float32)
    for i_c in range(0, NC, BC):
        o_c = i_c + tl.arange(0, BC)
        p_k = tl.make_block_ptr(k + (boc * H + i_h) * K, (K, TC), (1, H*K), (0, i_c), (BK, BC), (0, 1))
        p_v = tl.make_block_ptr(v + (boc * H + i_h) * V, (V, TC), (1, H*V), (i_v * BV, i_c), (BV, BC), (0, 1))
        # [BK, BC]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BV, BC]
        b_v = tl.load(p_v, boundary_check=(0, 1))

        # [G, BC]
        b_s = tl.dot(b_q, b_k)
        b_p = exp(b_s - b_lse[:, None])
        b_p = tl.where((o_c < NC)[None, :], b_p, 0)

        # [G, BV] @ [BV, BC] -> [G, BC]
        b_dp = tl.dot(b_do, b_v)
        b_ds = b_p * (b_dp.to(tl.float32) - b_delta[:, None])
        # [G, BC] @ [BC, BK] -> [G, BK]
        b_dq += tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k))
    b_dq *= scale
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))


# ============================================================================
# Call function: parallel_nsa_compression_bwd (partial - only dq kernel)
# ============================================================================

def parallel_nsa_compression_bwd_dq(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    do: torch.Tensor,
    block_size: int = 64,
    scale: float = None,
    cu_seqlens: torch.LongTensor | None = None,
    token_indices: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
):
    """
    Backward pass for parallel_nsa_compression - only computing dq
    """
    B, T, HQ, K, V = *q.shape, v.shape[-1]
    TC = k.shape[1]
    H = k.shape[2]
    G = HQ // H
    BC = BS = block_size
    BK = max(triton.next_power_of_2(K), 16)
    BV = min(128, max(triton.next_power_of_2(v.shape[-1]), 16))
    NV = triton.cdiv(V, BV)
    if cu_seqlens is not None:
        chunk_offsets = prepare_chunk_offsets(cu_seqlens, BS)
        if chunk_indices is None:
            chunk_indices = prepare_chunk_indices(chunk_offsets, BC)
        NC = len(chunk_indices)
    else:
        chunk_indices, chunk_offsets = None, None
        NC = triton.cdiv(triton.cdiv(T, BS), BC)

    delta = parallel_attn_bwd_preprocess(o, do)

    dq = torch.empty(NV, *q.shape, dtype=q.dtype if NV == 1 else torch.float, device=q.device)
    grid = (T, NV, B * H)
    parallel_nsa_compression_bwd_kernel_dq[grid](
        q=q,
        k=k,
        v=v,
        lse=lse,
        delta=delta,
        do=do,
        dq=dq,
        scale=scale,
        cu_seqlens=cu_seqlens,
        token_indices=token_indices,
        chunk_offsets=chunk_offsets,
        T=T,
        B=B,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        V=V,
        BC=BC,
        BS=BS,
        BK=BK,
        BV=BV,
    )
    dq = dq.sum(0)
    
    return dq


# ============================================================================
# Test case
# ============================================================================

def test_parallel_nsa_compression_bwd_kernel_dq():
    """
    Test case for parallel_nsa_compression_bwd_kernel_dq
    Using official test config: (B=3, T=1024, H=2, HQ=32, D=60, S=16, block_size=32, scale=0.1, float16)
    """
    # Official test config
    B = 3
    T = 1024
    H = 2
    HQ = 32
    D = 60
    S = 16
    block_size = 32
    scale = 0.1
    
    device = torch.device('npu') if torch.npu.is_available() else torch.device('cpu')
    
    # Load input tensors from pt file
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/parallel_nsa_compression_bwd_kernel_dq'
    inputs_dict = torch.load(os.path.join(load_dir, 'inputs.pt'))
    
    q = inputs_dict['q'].to(device)
    k_cmp = inputs_dict['k_cmp'].to(device)
    v_cmp = inputs_dict['v_cmp'].to(device)
    o_cmp = inputs_dict['o_cmp'].to(device)
    lse_cmp = inputs_dict['lse_cmp'].to(device)
    do = inputs_dict['do'].to(device)
    block_size = inputs_dict['block_size']
    scale = inputs_dict['scale']
    
    print(f"Testing parallel_nsa_compression_bwd_kernel_dq with:")
    print(f"  B={B}, T={T}, H={H}, HQ={HQ}, D={D}")
    print(f"  S={S}, block_size={block_size}, scale={scale}")
    print(f"  device={device}")
    print(f"  q shape: {q.shape}")
    print(f"  k_cmp shape: {k_cmp.shape}")
    print(f"  v_cmp shape: {v_cmp.shape}")
    print(f"  o_cmp shape: {o_cmp.shape}")
    print(f"  lse_cmp shape: {lse_cmp.shape}")
    print(f"  do shape: {do.shape}")
    
    # For non-varlen case
    cu_seqlens = None
    token_indices = None
    chunk_indices = None
    
    # Run the kernel on NPU
    print("\nExecuting kernel on NPU...")
    dq_npu = parallel_nsa_compression_bwd_dq(
        q=q,
        k=k_cmp,
        v=v_cmp,
        o=o_cmp,
        lse=lse_cmp,
        do=do,
        block_size=block_size,
        scale=scale,
        cu_seqlens=cu_seqlens,
        token_indices=token_indices,
        chunk_indices=chunk_indices,
    )
    torch.npu.synchronize()
    
    # Load expected output tensors from pt file
    outputs_dict = torch.load(os.path.join(load_dir, 'outputs.pt'))
    
    # Convert expected outputs to NPU for comparison
    dq_gpu_fp32 = outputs_dict['dq_fp32'].to(device)
    dq_gpu = outputs_dict['dq'].to(device)
    
    # Compare outputs with expected values using NPU/GPU/GPU_FP32 three-way comparison
    print("\n" + "=" * 60)
    print("Comparing dq outputs...")
    print("=" * 60)
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(dq_npu, dq_gpu_fp32)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(dq_gpu, dq_gpu_fp32)
    print(f"dq 最大相对误差比: {relative_max_npu / relative_max_gpu:.4f}")
    print(f"dq 平均相对误差比: {relative_mean_npu / relative_mean_gpu:.4f}")
    print(f"dq 均方根误差比: {rmse_npu / rmse_gpu:.4f}")
    assert relative_max_npu / relative_max_gpu <= 2, "dq 最大相对误差比 check failed"
    assert relative_mean_npu / relative_mean_gpu <= 1.2, "dq 平均相对误差比 check failed"
    assert rmse_npu / rmse_gpu <= 1.2, "dq 均方根误差比 check failed"
    
    print("\n" + "=" * 60)
    print("All output comparisons passed!")
    print("=" * 60)
    
    print(f"\nOutput shapes:")
    print(f"  dq shape: {dq_npu.shape}")
    print(f"\nTest passed!")
    
    return dq_npu


# ============================================================================
# Profiling test
# ============================================================================

def profile_parallel_nsa_compression_bwd_kernel_dq():
    """
    Profiling test for parallel_nsa_compression_bwd_kernel_dq
    """
    # Official test config
    B = 3
    T = 1024
    H = 2
    HQ = 32
    D = 60
    S = 16
    block_size = 32
    scale = 0.1
    
    device = torch.device('npu') if torch.npu.is_available() else torch.device('cpu')
    
    # Load input tensors from pt file
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/parallel_nsa_compression_bwd_kernel_dq'
    inputs_dict = torch.load(os.path.join(load_dir, 'inputs.pt'))
    
    q = inputs_dict['q'].to(device)
    k_cmp = inputs_dict['k_cmp'].to(device)
    v_cmp = inputs_dict['v_cmp'].to(device)
    o_cmp = inputs_dict['o_cmp'].to(device)
    lse_cmp = inputs_dict['lse_cmp'].to(device)
    do = inputs_dict['do'].to(device)
    block_size = inputs_dict['block_size']
    scale = inputs_dict['scale']
    
    cu_seqlens = None
    token_indices = None
    chunk_indices = None
    
    print(f"\nProfiling parallel_nsa_compression_bwd_kernel_dq...")
    print(f"  B={B}, T={T}, H={H}, HQ={HQ}, D={D}")
    
    # Warmup
    print("\nWarmup run...")
    for _ in range(5):
        _ = parallel_nsa_compression_bwd_dq(
            q=q,
            k=k_cmp,
            v=v_cmp,
            o=o_cmp,
            lse=lse_cmp,
            do=do,
            block_size=block_size,
            scale=scale,
            cu_seqlens=cu_seqlens,
            token_indices=token_indices,
            chunk_indices=chunk_indices,
        )
        if device.type == 'npu':
            torch.npu.synchronize()
    print("Warmup completed.")
    
    # Profiling
    print("\nStarting profiling...")
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
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result_dir_parallel_nsa_compression_bwd_kernel_dq")
    ) as prof:
        prof.start()
        for i in range(35):
            _ = parallel_nsa_compression_bwd_dq(
                q=q,
                k=k_cmp,
                v=v_cmp,
                o=o_cmp,
                lse=lse_cmp,
                do=do,
                block_size=block_size,
                scale=scale,
                cu_seqlens=cu_seqlens,
                token_indices=token_indices,
                chunk_indices=chunk_indices,
            )
            torch.npu.synchronize()
            prof.step()
        prof.stop()
    
    print("Profiling completed.")
    print(f"Profiling results saved to: ./result_dir_parallel_nsa_compression_bwd_kernel_dq")


if __name__ == "__main__":
    print("=" * 80)
    print("Testing parallel_nsa_compression_bwd_kernel_dq")
    print("Using official test config: (B=3, T=1024, H=2, HQ=32, D=60, S=16, block_size=32, scale=0.1, float16)")
    print("=" * 80)
    
    # Run basic test
    test_parallel_nsa_compression_bwd_kernel_dq()
    
    # Run profiling test
    profile_parallel_nsa_compression_bwd_kernel_dq()
    
    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)
