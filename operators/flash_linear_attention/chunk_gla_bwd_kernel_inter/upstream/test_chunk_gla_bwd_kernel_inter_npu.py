# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch_npu
import triton
import triton.language as tl
from typing import Optional

# ============== Helper Functions ==============

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
def chunk_gla_bwd_kernel_inter(
    q,
    k,
    v,
    g,
    h,
    do,
    dh,
    dq,
    dk,
    dq2,
    dk2,
    dg,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
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
    o_k = i_k * BK + tl.arange(0, BK)
    m_k = o_k < K

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    g += (bos * H + i_h) * K
    h += (i_tg * H + i_h) * K*V
    do += (bos * H + i_h) * V
    dh += (i_tg * H + i_h) * K*V
    dq += (bos * H + i_h) * K
    dk += (bos * H + i_h) * K
    dq2 += (bos * H + i_h) * K
    dk2 += (bos * H + i_h) * K
    dg += (bos * H + i_h) * K

    p_gk = tl.make_block_ptr(g, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    b_gk = tl.load(p_gk, boundary_check=(0, 1))
    p_gn = g + (min(T, i_t * BT + BT) - 1) * H*K + o_k
    b_gn = tl.load(p_gn, mask=m_k, other=0)
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_dgk = tl.zeros([BK], dtype=tl.float32)

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_do = tl.make_block_ptr(do, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        p_dh = tl.make_block_ptr(dh, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        # [BV, BK]
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_dh = tl.load(p_dh, boundary_check=(0, 1))

        # [BK]
        b_dgk += tl.sum(b_h * b_dh, axis=0)
        # [BT, BK]
        b_dq += tl.dot(b_do, b_h.to(b_do.dtype))
        b_dk += tl.dot(b_v, b_dh.to(b_v.dtype))

    b_dgk *= tl.exp2(b_gn)
    b_dq *= scale
    b_dq = b_dq * tl.exp2(b_gk)
    b_dk = b_dk * tl.exp2(b_gn[None, :] - b_gk)
    p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dq = tl.make_block_ptr(dq, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_dgk += tl.sum(b_dk * b_k, axis=0)
    b_dq += tl.load(p_dq, boundary_check=(0, 1))
    b_dk += tl.load(p_dk, boundary_check=(0, 1))
    b_dg = b_q * b_dq - b_k * b_dk
    # tl.debug_barrier()
    b_dg = b_dg - tl.cumsum(b_dg, axis=0) + tl.sum(b_dg, axis=0)[None, :] + b_dgk[None, :]
    # Buggy due to strange triton compiler issue.
    # m_s = tl.where(tl.arange(0, BT)[:, None] <= tl.arange(0, BT)[None, :], 1., 0.)
    # b_dg = tl.dot(m_s, b_dg, allow_tf32=False) + b_dgk[None, :]
    p_dq = tl.make_block_ptr(dq2, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk2, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dg = tl.make_block_ptr(dg, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0, 1))


# ============== Call Function ==============

def chunk_gla_bwd_dqkg(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
):
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    dg = torch.empty_like(g)
    dq2 = torch.empty_like(dq)
    dk2 = torch.empty_like(dk)
    
    # Use fixed config values
    BK = 32
    BV = 64
    
    grid = (triton.cdiv(K, BK), NT, B * H)
    chunk_gla_bwd_kernel_inter[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        h=h,
        do=do,
        dh=dh,
        dq=dq,
        dk=dk,
        dq2=dq2,
        dk2=dk2,
        dg=dg,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
    )
    return dq2, dk2, dg


# ============== Test Case ==============

def test_chunk_gla_bwd_kernel_inter():
    """Test chunk_gla_bwd_kernel_inter with profiling"""
    
    # Config with large shapes for performance testing
    B = 2
    T = 512
    H = 8
    K = 64
    V = 64
    chunk_size = 32
    scale = K ** -0.5
    
    print(f"Test config: B={B}, T={T}, H={H}, K={K}, V={V}, chunk_size={chunk_size}")
    
    # Create input tensors on NPU
    device = torch.device('npu:0')
    torch.npu.set_device(device)
    
    q = torch.randn(B, T, H, K, dtype=torch.float32, device=device)
    k = torch.randn(B, T, H, K, dtype=torch.float32, device=device)
    v = torch.randn(B, T, H, V, dtype=torch.float32, device=device)
    g = torch.randn(B, T, H, K, dtype=torch.float32, device=device)
    do = torch.randn(B, T, H, V, dtype=torch.float32, device=device)
    
    # Create h and dh tensors (hidden state and gradient)
    NT = (T + chunk_size - 1) // chunk_size
    h = torch.randn(NT * B, H, K, V, dtype=torch.float32, device=device)
    dh = torch.randn(NT * B, H, K, V, dtype=torch.float32, device=device)
    
    # Create dq and dk tensors (intra-chunk gradients)
    dq = torch.randn(B, T, H, K, dtype=torch.float32, device=device)
    dk = torch.randn(B, T, H, K, dtype=torch.float32, device=device)
    
    print(f"Input shapes: q={q.shape}, k={k.shape}, v={v.shape}, g={g.shape}, do={do.shape}, h={h.shape}, dh={dh.shape}, dq={dq.shape}, dk={dk.shape}")
    
    # Warmup runs
    print("Running warmup...")
    for _ in range(5):
        _ = chunk_gla_bwd_dqkg(
            q=q, k=k, v=v, h=h, g=g, do=do, dh=dh, dq=dq, dk=dk, scale=scale, chunk_size=chunk_size
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
            _ = chunk_gla_bwd_dqkg(
                q=q, k=k, v=v, h=h, g=g, do=do, dh=dh, dq=dq, dk=dk, scale=scale, chunk_size=chunk_size
            )
            torch.npu.synchronize()
            prof.step()
        prof.stop()
    
    print("Profiling completed!")
    print(f"Results saved to: ./result_dir_original")
    
    # Verify output
    dq2, dk2, dg = chunk_gla_bwd_dqkg(
        q=q, k=k, v=v, h=h, g=g, do=do, dh=dh, dq=dq, dk=dk, scale=scale, chunk_size=chunk_size
    )
    print(f"Output dq2 shape: {dq2.shape}, dtype: {dq2.dtype}")
    print(f"Output dk2 shape: {dk2.shape}, dtype: {dk2.dtype}")
    print(f"Output dg shape: {dg.shape}, dtype: {dg.dtype}")
    print(f"dq2 contains NaN: {torch.isnan(dq2).any().item()}")
    print(f"dq2 contains Inf: {torch.isinf(dq2).any().item()}")
    print(f"dk2 contains NaN: {torch.isnan(dk2).any().item()}")
    print(f"dk2 contains Inf: {torch.isinf(dk2).any().item()}")
    print(f"dg contains NaN: {torch.isnan(dg).any().item()}")
    print(f"dg contains Inf: {torch.isinf(dg).any().item()}")
    
    return dq2, dk2, dg


if __name__ == "__main__":
    test_chunk_gla_bwd_kernel_inter()
