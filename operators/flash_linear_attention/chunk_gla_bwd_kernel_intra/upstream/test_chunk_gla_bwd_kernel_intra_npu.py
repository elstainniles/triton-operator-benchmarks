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
def chunk_gla_bwd_kernel_intra(
    q,
    k,
    g,
    dA,
    dq,
    dk,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_kc, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    i_k, i_i = i_kc // NC, i_kc % NC
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    else:
        bos, eos = i_b * T, i_b * T + T
    T = eos - bos
    if i_t * BT + i_i * BC >= T:
        return

    o_k = i_k * BK + tl.arange(0, BK)
    m_k = o_k < K

    p_g = tl.make_block_ptr(g + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    # [BC, BK]
    b_g = tl.load(p_g, boundary_check=(0, 1))

    b_dq = tl.zeros([BC, BK], dtype=tl.float32)
    if i_i > 0:
        p_gn = g + (bos + i_t * BT + i_i * BC) * H*K + i_h*K + o_k
        # [BK,]
        b_gn = tl.load(p_gn, mask=m_k, other=0)
        for i_j in range(0, i_i):
            p_k = tl.make_block_ptr(k+(bos*H+i_h)*K, (T, K), (H*K, 1), (i_t*BT+i_j*BC, i_k * BK), (BC, BK), (1, 0))
            p_gk = tl.make_block_ptr(g+(bos*H+i_h)*K, (T, K), (H*K, 1), (i_t*BT+i_j*BC, i_k * BK), (BC, BK), (1, 0))
            p_dA = tl.make_block_ptr(dA+(bos*H+i_h)*BT, (T, BT), (H*BT, 1), (i_t*BT+i_i*BC, i_j * BC), (BC, BC), (1, 0))
            # [BC, BK]
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_gk = tl.load(p_gk, boundary_check=(0, 1))
            b_kg = (b_k * tl.exp2(b_gn[None, :] - b_gk)).to(tl.float32)
            # [BC, BC]
            b_dA = tl.load(p_dA, boundary_check=(0, 1))

            b_dq += tl.dot(b_dA, b_kg)
        b_dq *= tl.exp2(b_g - b_gn[None, :])
    o_i = tl.arange(0, BC)
    m_dA = (i_t * BT + i_i * BC + tl.arange(0, BC)) < T
    o_dA = bos*H*BT + (i_t * BT + i_i * BC + tl.arange(0, BC)) * H*BT + i_h * BT + i_i * BC
    p_kj = k + (bos + i_t * BT + i_i * BC) * H*K + i_h * K + o_k
    p_gkj = g + (bos + i_t * BT + i_i * BC) * H*K + i_h * K + o_k
    p_dq = tl.make_block_ptr(dq + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))

    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        # [BC,]
        b_dA = tl.load(dA + o_dA + j, mask=m_dA, other=0)
        # [BK,]
        b_kj = tl.load(p_kj, mask=m_k, other=0).to(tl.float32)
        b_gkj = tl.load(p_gkj, mask=m_k, other=0).to(tl.float32)
        # [BC, BK]
        m_i = o_i[:, None] >= j
        # [BC, BK]
        # (SY 09/17) important to not use bf16 here to have a good precision.
        b_dq += tl.where(m_i, b_dA[:, None] * b_kj[None, :] * tl.exp2(b_g - b_gkj[None, :]), 0.)
        p_kj += H*K
        p_gkj += H*K
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))

    tl.debug_barrier()
    # [BC, BK]
    b_dk = tl.zeros([BC, BK], dtype=tl.float32)

    NC = min(NC, tl.cdiv(T - i_t * BT, BC))
    if i_i < NC - 1:
        p_gn = g + (bos + min(i_t * BT + i_i * BC + BC, T) - 1) * H*K + i_h * K + o_k

        # [BK,]
        b_gn = tl.load(p_gn, mask=m_k, other=0)
        for i_j in range(i_i + 1, NC):
            p_q = tl.make_block_ptr(q + (bos*H+i_h)*K, (T, K), (H*K, 1), (i_t*BT+i_j*BC, i_k*BK), (BC, BK), (1, 0))
            p_gq = tl.make_block_ptr(g + (bos*H+i_h)*K, (T, K), (H*K, 1), (i_t*BT+i_j*BC, i_k*BK), (BC, BK), (1, 0))
            p_dA = tl.make_block_ptr(dA + (bos*H+i_h)*BT, (BT, T), (1, H*BT), (i_i*BC, i_t*BT + i_j*BC), (BC, BC), (0, 1))

            o_j = i_t * BT + i_j * BC + o_i
            m_j = o_j < T
            # [BC, BK]
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_gq = tl.load(p_gq, boundary_check=(0, 1))
            b_qg = (b_q * tl.where(m_j[:, None], tl.exp2(b_gq - b_gn[None, :]), 0)).to(tl.float32)
            # [BC, BC]
            b_dA = tl.load(p_dA, boundary_check=(0, 1))
            # [BC, BK]
            # (SY 09/17) important to not use bf16 here to have a good precision.
            b_dk += tl.dot(b_dA, b_qg)
        b_dk *= tl.exp2(b_gn[None, :] - b_g)
    o_dA = bos*H*BT + (i_t * BT + i_i * BC) * H*BT + i_h * BT + i_i * BC + tl.arange(0, BC)
    p_qj = q + (bos + i_t * BT + i_i * BC) * H*K + i_h * K + o_k
    p_gqj = g + (bos + i_t * BT + i_i * BC) * H*K + i_h * K + o_k
    p_dk = tl.make_block_ptr(dk + (bos*H+i_h)*K, (T, K), (H*K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        # [BC,]
        b_dA = tl.load(dA + o_dA + j * H*BT)
        # [BK,]
        b_qj = tl.load(p_qj, mask=m_k, other=0).to(tl.float32)
        b_gqj = tl.load(p_gqj, mask=m_k, other=0).to(tl.float32)
        # [BC, BK]
        m_i = o_i[:, None] <= j
        b_dk += tl.where(m_i, b_dA[:, None] * b_qj[None, :] * tl.exp2(b_gqj[None, :] - b_g), 0.)
        p_qj += H*K
        p_gqj += H*K
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))


# ============== Call Function ==============

def chunk_gla_bwd_dqk_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    dA: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
):
    B, T, H, K = q.shape
    BT = chunk_size
    BC = min(16, BT)
    BK = min(64, triton.next_power_of_2(K))

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    NC = triton.cdiv(BT, BC)
    NK = triton.cdiv(K, BK)

    dq = torch.empty_like(q, dtype=torch.float)
    dk = torch.empty_like(k, dtype=torch.float)
    grid = (NK * NC, NT, B * H)
    chunk_gla_bwd_kernel_intra[grid](
        q=q,
        k=k,
        g=g,
        dA=dA,
        dq=dq,
        dk=dk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BC=BC,
        BK=BK,
        NC=NC,
    )
    return dq, dk


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

def test_chunk_gla_bwd_kernel_intra():
    """Test chunk_gla_bwd_kernel_intra with profiling"""
    
    # Load input tensors from pt files
    load_dir = "/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_gla_bwd_kernel_intra"
    inputs_dict = torch.load(os.path.join(load_dir, "inputs.pt"))
    
    device = torch.device('npu:0')
    torch.npu.set_device(device)
    
    # Convert loaded tensors to NPU
    q = inputs_dict['q'].to(device)
    k = inputs_dict['k'].to(device)
    g = inputs_dict['g'].to(device)
    dA = inputs_dict['dA'].to(device)
    cu_seqlens = inputs_dict['cu_seqlens']
    chunk_size = inputs_dict['chunk_size']
    
    print(f"Input shapes: q={q.shape}, k={k.shape}, g={g.shape}, dA={dA.shape}")
    
    # Execute the kernel once
    print("\nExecuting kernel...")
    dq_npu, dk_npu = chunk_gla_bwd_dqk_intra(
        q=q, k=k, g=g, dA=dA, chunk_size=chunk_size
    )
    
    # Load expected output tensors from pt file
    outputs_dict = torch.load(os.path.join(load_dir, "outputs.pt"))
    dq_gpu_fp64 = outputs_dict['dq_fp64'].to(device)
    dk_gpu_fp64 = outputs_dict['dk_fp64'].to(device)
    dq_gpu = outputs_dict['dq'].to(device)
    dk_gpu = outputs_dict['dk'].to(device)
    
    # Compare outputs with expected values
    print("\nComparing dq outputs with expected values...")
    relative_max_npu_dq, relative_mean_npu_dq, rmse_npu_dq = cal_precision(dq_npu, dq_gpu_fp64)
    relative_max_gpu_dq, relative_mean_gpu_dq, rmse_gpu_dq = cal_precision(dq_gpu, dq_gpu_fp64)
    print(f"dq 最大相对误差比 :{relative_max_npu_dq / relative_max_gpu_dq}, 平均相对误差比 :{relative_mean_npu_dq / relative_mean_gpu_dq}, 均方根误差比 :{rmse_npu_dq / rmse_gpu_dq}")
    assert relative_max_npu_dq / relative_max_gpu_dq <= 2, "dq 最大相对误差比 check failed"
    assert relative_mean_npu_dq / relative_mean_gpu_dq <= 1.2, "dq 平均相对误差比 check failed"
    assert rmse_npu_dq / rmse_gpu_dq <= 1.2, "dq 均方根误差比 check failed"
    
    print("\nComparing dk outputs with expected values...")
    relative_max_npu_dk, relative_mean_npu_dk, rmse_npu_dk = cal_precision(dk_npu, dk_gpu_fp64)
    relative_max_gpu_dk, relative_mean_gpu_dk, rmse_gpu_dk = cal_precision(dk_gpu, dk_gpu_fp64)
    print(f"dk 最大相对误差比 :{relative_max_npu_dk / relative_max_gpu_dk}, 平均相对误差比 :{relative_mean_npu_dk / relative_mean_gpu_dk}, 均方根误差比 :{rmse_npu_dk / rmse_gpu_dk}")
    assert relative_max_npu_dk / relative_max_gpu_dk <= 2, "dk 最大相对误差比 check failed"
    assert relative_mean_npu_dk / relative_mean_gpu_dk <= 1.2, "dk 平均相对误差比 check failed"
    assert rmse_npu_dk / rmse_gpu_dk <= 1.2, "dk 均方根误差比 check failed"
    print("Output comparison passed!")
    
    # Warmup runs
    print("\nRunning warmup...")
    for _ in range(5):
        _ = chunk_gla_bwd_dqk_intra(
            q=q, k=k, g=g, dA=dA, chunk_size=chunk_size
        )
        torch.npu.synchronize()
    
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
            wait=1, warmup=1, active=30, repeat=1, skip_first=1
        ),
        experimental_config=experimental_config,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
            "./result_dir_original"
        )
    ) as prof:
        prof.start()
        for i in range(35):
            _ = chunk_gla_bwd_dqk_intra(
                q=q, k=k, g=g, dA=dA, chunk_size=chunk_size
            )
            torch.npu.synchronize()
            prof.step()
        prof.stop()
    
    print("\nProfiling completed!")
    print(f"Results saved to: ./result_dir_original")
    
    print(f"\nOutput dq shape: {dq_npu.shape}, dtype: {dq_npu.dtype}")
    print(f"Output dk shape: {dk_npu.shape}, dtype: {dk_npu.dtype}")
    print(f"dq contains NaN: {torch.isnan(dq_npu).any().item()}")
    print(f"dq contains Inf: {torch.isinf(dq_npu).any().item()}")
    print(f"dk contains NaN: {torch.isnan(dk_npu).any().item()}")
    print(f"dk contains Inf: {torch.isinf(dk_npu).any().item()}")
    
    return dq_npu, dk_npu


if __name__ == "__main__":
    test_chunk_gla_bwd_kernel_intra()
