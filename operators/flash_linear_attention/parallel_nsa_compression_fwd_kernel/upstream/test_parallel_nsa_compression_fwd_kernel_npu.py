# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional
import os
import functools
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import torch_npu


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
# Kernel definition: parallel_nsa_compression_fwd_kernel
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
@triton.jit
def parallel_nsa_compression_fwd_kernel(
    q,
    k,
    v,
    o,
    lse,
    scale,
    cu_seqlens,
    token_indices,
    chunk_offsets,
    T,
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

    if IS_VARLEN:
        i_n, i_t = tl.load(token_indices + i_t * 2).to(tl.int32), tl.load(token_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        boc = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_b * T, i_b * T + T
        boc = i_b * tl.cdiv(T, BS)

    p_q = tl.make_block_ptr(q + (bos + i_t) * HQ*K, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0))

    # the Q block is kept in the shared memory throughout the whole kernel
    # [G, BK]
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)

    # the number of compression representations in total
    TC = tl.cdiv(T, BS)
    # the number of compression representations required to iterate over
    # incomplete compression blocks are not included
    NC = (i_t + 1) // BS

    p_o = tl.make_block_ptr(o + (bos + i_t) * HQ*V, (HQ, V), (V, 1), (i_h * G, i_v * BV), (G, BV), (1, 0))
    # [G, BV]
    b_o = tl.zeros([G, BV], dtype=tl.float32)
    # max scores for the current block
    b_m = tl.full([G], float('-inf'), dtype=tl.float32)
    # lse = log(acc) + m
    b_acc = tl.zeros([G], dtype=tl.float32)

    for i_c in range(0, NC, BC):
        o_c = i_c + tl.arange(0, BC)

        p_k = tl.make_block_ptr(k + (boc * H + i_h) * K, (K, TC), (1, H*K), (0, i_c), (BK, BC), (0, 1))
        p_v = tl.make_block_ptr(v + (boc * H + i_h) * V, (TC, V), (H*V, 1), (i_c, i_v * BV), (BC, BV), (1, 0))
        # [BK, BC]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BC, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [G, BC]
        b_s = tl.dot(b_q, b_k)
        b_s = tl.where((o_c < NC)[None, :], b_s, float('-inf'))

        # [G]
        b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
        b_r = exp(b_mp - b_m)
        # [G, BC]
        b_p = exp(b_s - b_m[:, None])
        # [G]
        b_acc = b_acc * b_r + tl.sum(b_p, 1)

        # [G, BV]
        b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_q.dtype), b_v)

        b_mp = b_m
    if NC == 0:
        b_lse = tl.zeros([G], dtype=tl.float32)
    else:
        b_o = b_o / b_acc[:, None]
        b_lse = b_m + log(b_acc)

    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    if i_v == 0:
        tl.store(lse + (bos + i_t) * HQ + i_h * G + tl.arange(0, G), b_lse.to(lse.dtype.element_ty))


# ============================================================================
# Call function: parallel_nsa_compression_fwd
# ============================================================================

def parallel_nsa_compression_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    token_indices: torch.LongTensor | None = None,
):
    B, T, HQ, K, V = *q.shape, v.shape[-1]
    H = k.shape[2]
    G = HQ // H
    BC = BS = block_size
    if check_shared_mem('hopper', q.device.index):
        BK = min(256, triton.next_power_of_2(K))
        BV = min(256, triton.next_power_of_2(V))
    else:
        BK = min(128, triton.next_power_of_2(K))
        BV = min(128, triton.next_power_of_2(V))
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    assert NK == 1, "The key dimension can not be larger than 256"

    chunk_offsets = prepare_chunk_offsets(cu_seqlens, BS) if cu_seqlens is not None else None

    grid = (T, NV, B * H)
    o = torch.empty(B, T, HQ, V, dtype=v.dtype, device=q.device)
    lse = torch.empty(B, T, HQ, dtype=torch.float, device=q.device)

    parallel_nsa_compression_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        o=o,
        lse=lse,
        scale=scale,
        cu_seqlens=cu_seqlens,
        token_indices=token_indices,
        chunk_offsets=chunk_offsets,
        T=T,
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
    return o, lse


# ============================================================================
# Test case
# ============================================================================

def test_parallel_nsa_compression_fwd_kernel():
    """
    Test case for parallel_nsa_compression_fwd_kernel
    Using large shapes for performance testing
    """
    # Configuration with large shapes
    B = 2  # batch size
    T = 1024  # sequence length
    H = 8  # number of heads for K/V
    HQ = 32  # number of heads for Q (with GQA, HQ = H * G)
    K = 128  # key dimension
    V = 128  # value dimension
    block_size = 64  # compression block size
    scale = K ** -0.5
    
    device = torch.device('npu:0') if torch.npu.is_available() else torch.device('cpu')
    dtype = torch.float16
    
    # Load input tensors from pt file
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/parallel_nsa_compression_fwd_kernel'
    q = torch.load(os.path.join(load_dir, 'q.pt')).to(device)
    k = torch.load(os.path.join(load_dir, 'k.pt')).to(device)
    v = torch.load(os.path.join(load_dir, 'v.pt')).to(device)
    print(f"Input tensors loaded from {load_dir}")
    
    # For non-varlen case, cu_seqlens and token_indices are None
    cu_seqlens = None
    token_indices = None
    
    print(f"Testing parallel_nsa_compression_fwd_kernel with:")
    print(f"  B={B}, T={T}, H={H}, HQ={HQ}, K={K}, V={V}")
    print(f"  block_size={block_size}, scale={scale:.6f}")
    print(f"  device={device}, dtype={dtype}")
    print(f"  q shape: {q.shape}")
    print(f"  k shape: {k.shape}")
    print(f"  v shape: {v.shape}")
    
    # Run the kernel
    o, lse = parallel_nsa_compression_fwd(
        q=q,
        k=k,
        v=v,
        block_size=block_size,
        scale=scale,
        cu_seqlens=cu_seqlens,
        token_indices=token_indices,
    )
    
    # Load expected output tensors from pt file
    o_expected = torch.load(os.path.join(load_dir, 'o.pt')).to(device)
    lse_expected = torch.load(os.path.join(load_dir, 'lse.pt')).to(device)
    print(f"Expected output tensors loaded from {load_dir}")
    
    # Compare outputs with expected values
    print(f"\nComparing outputs with expected values...")
    validate_cmp("float16", o, o_expected)
    validate_cmp("float16", lse, lse_expected)
    print(f"Output comparison passed!")
    
    print(f"\nOutput shapes:")
    print(f"  o shape: {o.shape}")
    print(f"  lse shape: {lse.shape}")
    print(f"\nTest passed!")
    
    return o, lse


# ============================================================================
# Profiling test
# ============================================================================

def profile_parallel_nsa_compression_fwd_kernel():
    """
    Profiling test for parallel_nsa_compression_fwd_kernel
    """
    # Configuration with large shapes
    B = 2
    T = 1024
    H = 8
    HQ = 32
    K = 128
    V = 128
    block_size = 64
    scale = K ** -0.5
    
    device = torch.device('npu:0') if torch.npu.is_available() else torch.device('cpu')
    dtype = torch.float16
    
    # Load input tensors from pt file
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/parallel_nsa_compression_fwd_kernel'
    q = torch.load(os.path.join(load_dir, 'q.pt')).to(device)
    k = torch.load(os.path.join(load_dir, 'k.pt')).to(device)
    v = torch.load(os.path.join(load_dir, 'v.pt')).to(device)
    
    cu_seqlens = None
    token_indices = None
    
    print(f"\nProfiling parallel_nsa_compression_fwd_kernel...")
    print(f"  B={B}, T={T}, H={H}, HQ={HQ}, K={K}, V={V}")
    
    # Warmup
    for _ in range(5):
        o, lse = parallel_nsa_compression_fwd(
            q=q,
            k=k,
            v=v,
            block_size=block_size,
            scale=scale,
            cu_seqlens=cu_seqlens,
            token_indices=token_indices,
        )
        if device.type == 'npu':
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
            o, lse = parallel_nsa_compression_fwd(
                q=q,
                k=k,
                v=v,
                block_size=block_size,
                scale=scale,
                cu_seqlens=cu_seqlens,
                token_indices=token_indices,
            )
            torch.npu.synchronize()
            prof.step()
        prof.stop()
    
    print(f"Profiling completed! Results saved to ./result_dir_original")


if __name__ == "__main__":
    print("=" * 80)
    print("Testing parallel_nsa_compression_fwd_kernel")
    print("=" * 80)
    
    # Run basic test
    test_parallel_nsa_compression_fwd_kernel()
    
    # Run profiling test
    profile_parallel_nsa_compression_fwd_kernel()
    
    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)
