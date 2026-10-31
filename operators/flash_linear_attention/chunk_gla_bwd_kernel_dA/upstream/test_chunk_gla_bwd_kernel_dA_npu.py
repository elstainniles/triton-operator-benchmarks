# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional
import torch
import torch_npu
import triton
import triton.language as tl
import os

# ============== Helper Functions ==============

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

def check_shared_mem(arch: str = None):
    """Check if shared memory is available"""
    return True


def autotune_cache_kwargs():
    """Autotune cache kwargs"""
    return {}


def fla_cache_autotune(*args, **kwargs):
    """FLA cache autotune decorator"""
    def decorator(func):
        return func
    return decorator


def input_guard(func):
    """Input guard decorator"""
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int, cu_seqlens_cpu: Optional[torch.Tensor] = None):
    """Prepare chunk indices for variable-length sequences"""
    if cu_seqlens is None:
        return None
    
    # Calculate number of chunks for each sequence
    chunk_indices = []
    for i in range(len(cu_seqlens) - 1):
        start = cu_seqlens[i].item()
        end = cu_seqlens[i + 1].item()
        seq_len = end - start
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        for j in range(num_chunks):
            chunk_indices.append([i, j])
    
    return torch.tensor(chunk_indices, dtype=torch.long, device=cu_seqlens.device)


# ============== Kernel Definition ==============

@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def chunk_gla_bwd_kernel_dA(
    v,
    do,
    dA,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    else:
        bos, eos = i_b * T, i_b * T + T
    T = eos - bos

    b_dA = tl.zeros([BT, BT], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        p_do = tl.make_block_ptr(do + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_v = tl.make_block_ptr(v + (bos*H + i_h) * V, (V, T), (1, H*V), (i_v * BV, i_t * BT), (BV, BT), (0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))

        b_dA += tl.dot(b_do, b_v)

    p_dA = tl.make_block_ptr(dA + (bos * H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]
    b_dA = tl.where(m_s, b_dA * scale, 0.)
    tl.store(p_dA, b_dA.to(p_dA.dtype.element_ty), boundary_check=(0, 1))


# ============== Call Function ==============

def chunk_gla_bwd_dA(
    v: torch.Tensor,
    do: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
):
    B, T, H, V = v.shape
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    BV = min(64, triton.next_power_of_2(V))

    dA = v.new_empty(B, T, H, BT, dtype=torch.float)
    grid = (NT, B * H)
    chunk_gla_bwd_kernel_dA[grid](
        v=v,
        do=do,
        dA=dA,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        V=V,
        BT=BT,
        BV=BV,
    )
    return dA


# ============== Test Case ==============

def test_chunk_gla_bwd_kernel_dA():
    """Test chunk_gla_bwd_kernel_dA with profiling"""
    
    # Config with large shapes for performance testing
    B = 2
    T = 1024
    H = 8
    V = 128
    chunk_size = 64
    scale = V ** -0.5
    
    print(f"Test config: B={B}, T={T}, H={H}, V={V}, chunk_size={chunk_size}")
    
    # Create input tensors on NPU
    device = torch.device('npu:0')
    torch.npu.set_device(device)
    
    input_dir = "/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_gla_bwd_kernel_dA"
    print(f"Loading inputs from: {input_dir}")
    
    v = torch.load(os.path.join(input_dir, "input_v.pt")).to(device)
    do = torch.load(os.path.join(input_dir, "input_do.pt")).to(device)

    dA = chunk_gla_bwd_dA(
        v=v, do=do, scale=scale, chunk_size=chunk_size
    )

    expected_dA = torch.load(os.path.join(input_dir, "output_dA.pt")).to(device)
    
    print(f"Input shapes: v={v.shape}, do={do.shape}")

    print("Comparing output with expected value...")
    validate_cmp("float16", dA, expected_dA)
    print("✓ Output verification passed!")
    
    # Warmup runs
    print("Running warmup...")
    for _ in range(5):
        _ = chunk_gla_bwd_dA(
            v=v, do=do, scale=scale, chunk_size=chunk_size
        )
        torch.npu.synchronize()
    
    # Profiling
    print("Starting profiling...")
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
            wait=1, warmup=1, active=30, repeat=1, skip_first=1
        ),
        experimental_config=experimental_config,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
            "./result_dir_original"
        )
    ) as prof:
        prof.start()
        for i in range(35):
            _ = chunk_gla_bwd_dA(
                v=v, do=do, scale=scale, chunk_size=chunk_size
            )
            torch.npu.synchronize()
            prof.step()
        prof.stop()
    
    print("Profiling completed!")
    print(f"Results saved to: ./result_dir_original")
    
    # Verify output
    dA = chunk_gla_bwd_dA(
        v=v, do=do, scale=scale, chunk_size=chunk_size
    )
    print(f"Output dA shape: {dA.shape}, dtype: {dA.dtype}")
    print(f"dA contains NaN: {torch.isnan(dA).any().item()}")
    print(f"dA contains Inf: {torch.isinf(dA).any().item()}")
    
    return dA


if __name__ == "__main__":
    test_chunk_gla_bwd_kernel_dA()
