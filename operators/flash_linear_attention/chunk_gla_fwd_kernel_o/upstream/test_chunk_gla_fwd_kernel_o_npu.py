# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch_npu
import triton
import triton.language as tl
from typing import Optional
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
def chunk_gla_fwd_kernel_o(
    q,
    v,
    g,
    h,
    o,
    A,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    TRANSPOSE_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        i_tg = i_t.to(tl.int64)
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = (i_b * NT + i_t).to(tl.int64)
        bos, eos = (i_b * T).to(tl.int64), (i_b * T + T).to(tl.int64)

    m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]

    q += (bos * H + i_h) * K
    g += (bos * HV + i_hv) * K
    v += (bos * HV + i_hv) * V
    o += (bos * HV + i_hv) * V
    h += (i_tg * HV + i_hv).to(tl.int64) * K * V
    A += (bos * HV + i_hv) * BT

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_g = tl.make_block_ptr(g, (T, K), (HV*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        if TRANSPOSE_STATE:
            p_h = tl.make_block_ptr(h, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
        else:
            p_h = tl.make_block_ptr(h, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))

        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BT, BK]
        b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
        # [BT, BK]
        b_qg = (b_q * tl.exp2(b_g)).to(b_q.dtype)
        b_h = tl.load(p_h, boundary_check=(0, 1))
        if i_k >= 0:
            if TRANSPOSE_STATE:
                b_o += tl.dot(b_qg, tl.trans(b_h).to(b_qg.dtype))
            else:
                b_o += tl.dot(b_qg, b_h.to(b_qg.dtype))
    b_o *= scale
    p_v = tl.make_block_ptr(v, (T, V), (HV*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o, (T, V), (HV*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_A = tl.make_block_ptr(A, (T, BT), (HV*BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    # [BT, BV]
    b_v = tl.load(p_v, boundary_check=(0, 1))
    # [BT, BT]
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_A = tl.where(m_s, b_A, 0.).to(b_v.dtype)
    b_o += tl.dot(b_A, b_v)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


# ============== Call Function ==============

def chunk_gla_fwd_o_gk(
    q: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    A: torch.Tensor,
    h: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
    transpose_state_layout: bool = False,
):
    B, T, H, K, HV, V = *q.shape, v.shape[2], v.shape[-1]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    # Please ensure zeros, since vllm will use padding v
    o = torch.zeros_like(v)
    
    # Use fixed config values
    BK = 64
    BV = 128
    
    grid = (triton.cdiv(V, BV), NT, B * HV)
    chunk_gla_fwd_kernel_o[grid](
        q=q,
        v=v,
        g=g,
        h=h,
        o=o,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        TRANSPOSE_STATE=transpose_state_layout,
    )
    return o


# ============== Test Case ==============

def test_chunk_gla_fwd_kernel_o():
    """Test chunk_gla_fwd_kernel_o with profiling"""
    
    # Load input tensors from pt files
    load_dir = "/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_gla_fwd_kernel_o_v2"
    
    device = torch.device('npu:0')
    inputs_dict = torch.load(os.path.join(load_dir, "inputs.pt"))
    
    q = inputs_dict["q"].to(device)
    v = inputs_dict["v"].to(device)
    g = inputs_dict["g"].to(device)
    A = inputs_dict["A"].to(device)
    h = inputs_dict["h"].to(device)
    scale = inputs_dict["scale"]
    chunk_size = inputs_dict["chunk_size"]
    
    print(f"Input shapes: q={q.shape}, v={v.shape}, g={g.shape}, A={A.shape}, h={h.shape}")
    
    # Execute the kernel once
    o = chunk_gla_fwd_o_gk(
        q=q, v=v, g=g, A=A, h=h, scale=scale, chunk_size=chunk_size
    )
    
    # Load expected output and compare
    expected_o = torch.load(os.path.join(load_dir, "outputs.pt"))["o"].to(device)
    
    print("Comparing output with expected value...")
    validate_cmp("float16", o, expected_o)
    torch.testing.assert_close(o, expected_o, atol=1e-3, rtol=1e-3)
    print("✓ Output verification passed!")
    
    # Warmup runs
    print("Running warmup...")
    for _ in range(5):
        _ = chunk_gla_fwd_o_gk(
            q=q, v=v, g=g, A=A, h=h, scale=scale, chunk_size=chunk_size
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
            _ = chunk_gla_fwd_o_gk(
                q=q, v=v, g=g, A=A, h=h, scale=scale, chunk_size=chunk_size
            )
            torch.npu.synchronize()
            prof.step()
        prof.stop()
    
    print("Profiling completed!")
    print(f"Results saved to: ./result_dir_original")
    
    print(f"Output shape: {o.shape}")
    print(f"Output dtype: {o.dtype}")
    print(f"Output contains NaN: {torch.isnan(o).any().item()}")
    print(f"Output contains Inf: {torch.isinf(o).any().item()}")
    
    return o


if __name__ == "__main__":
    test_chunk_gla_fwd_kernel_o()
