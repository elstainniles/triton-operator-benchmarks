# Test case for parallel_nsa_bwd_kernel_dkv
# Extracted from fla/ops/nsa/parallel.py

import torch
import torch_npu
import triton
import triton.language as tl
import os


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


# ============== Helper functions from fla/ops/utils/op.py ==============
@triton.jit
def exp(x): 
    return tl.exp(x.to(tl.float32))


# ============== Kernel definition ==============
@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4]
    ],
    key=['BS', 'BK', 'BV'],
)
@triton.jit(do_not_specialize=['T'])
def parallel_nsa_bwd_kernel_dkv(
    q,
    k,
    v,
    lse,
    delta,
    do,
    dk,
    dv,
    block_mask,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    M: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_s, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    all = B * T
    if IS_VARLEN:
        i_n, i_s = tl.load(chunk_indices + i_s * 2).to(tl.int32), tl.load(chunk_indices + i_s * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_s * BS, 0), (BS, BK), (1, 0))
    p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H*V, 1), (i_s * BS, i_v * BV), (BS, BV), (1, 0))
    p_dk = tl.make_block_ptr(dk + (i_v * all * H + bos * H + i_h) * K, (T, K), (H*K, 1), (i_s * BS, 0), (BS, BK), (1, 0))
    p_dv = tl.make_block_ptr(dv + (bos * H + i_h) * V, (T, V), (H*V, 1), (i_s * BS, i_v * BV), (BS, BV), (1, 0))

    # [BS, BK]
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_dk = tl.zeros([BS, BK], dtype=tl.float32)
    # [BS, BV]
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_dv = tl.zeros([BS, BV], dtype=tl.float32)

    for i in range(i_s * BS, T):
        b_m = tl.load(block_mask + (bos + i) * H*M + i_h * M + i_s)
        if b_m:
            p_q = tl.make_block_ptr(q + (bos + i) * HQ*K, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0))
            # [G, BK]
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_q = (b_q * scale).to(b_q.dtype)

            p_do = tl.make_block_ptr(do + (bos + i) * HQ*V, (HQ, V), (V, 1), (i_h * G, i_v * BV), (G, BV), (1, 0))
            p_lse = lse + (bos + i) * HQ + i_h * G + tl.arange(0, G)
            p_delta = delta + (bos + i) * HQ + i_h * G + tl.arange(0, G)
            # [G, BV]
            b_do = tl.load(p_do, boundary_check=(0, 1))
            # [G]
            b_lse = tl.load(p_lse)
            b_delta = tl.load(p_delta)
            # [BS, G]
            b_s = tl.dot(b_k, tl.trans(b_q))
            b_p = exp(b_s - b_lse[None, :])
            b_p = tl.where((i >= (i_s * BS + tl.arange(0, BS)))[:, None], b_p, 0)
            # [BS, G] @ [G, BV] -> [BS, BV]
            b_dv += tl.dot(b_p.to(b_do.dtype), b_do)
            # [BS, BV] @ [BV, G] -> [BS, G]
            b_dp = tl.dot(b_v, tl.trans(b_do))
            # [BS, G]
            b_ds = b_p * (b_dp - b_delta[None, :])
            # [BS, G] @ [G, BK] -> [BS, BK]
            b_dk += tl.dot(b_ds.to(b_q.dtype), b_q)

    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


# ============== Helper function for backward preprocess ==============
def parallel_attn_bwd_preprocess(o, do):
    """Preprocess for backward pass - compute delta"""
    # delta = sum(do * o, dim=-1)
    delta = (do * o).sum(dim=-1)
    return delta


# ============== Wrapper function ==============
def parallel_nsa_bwd_dkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    do: torch.Tensor,
    block_mask: torch.Tensor,
    block_size: int = 64,
    scale: float = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
):
    B, T, H, K, V = *k.shape, v.shape[-1]
    HQ = q.shape[2]
    G = HQ // H
    BS = block_size
    BK = max(triton.next_power_of_2(K), 16)
    BV = min(128, max(triton.next_power_of_2(v.shape[-1]), 16))
    NV = triton.cdiv(V, BV)
    
    delta = parallel_attn_bwd_preprocess(o, do)
    
    if cu_seqlens is not None:
        if chunk_indices is None:
            # Simplified version for testing
            NS = triton.cdiv(T, BS)
        else:
            NS = len(chunk_indices)
    else:
        NS = triton.cdiv(T, BS)

    dk = torch.empty(NV, *k.shape, dtype=k.dtype if NV == 1 else torch.float, device=q.device)
    dv = torch.empty(v.shape, dtype=v.dtype, device=q.device)

    grid = (NV, NS, B * H)
    parallel_nsa_bwd_kernel_dkv[grid](
        q=q,
        k=k,
        v=v,
        lse=lse,
        delta=delta,
        do=do,
        dk=dk,
        dv=dv,
        block_mask=block_mask,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        B=B,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        V=V,
        M=block_mask.shape[-1],
        BS=BS,
        BK=BK,
        BV=BV,
    )
    dk = dk.sum(0)
    return dk, dv


# ============== Test configuration ==============
def get_test_config():
    """Generate test configuration with large shapes"""
    # Large shape configuration for performance testing
    # Note: NPU has a limit on grid dimensions (max 65535 total blocks)
    # Grid is (NV, NS, B * H), so we need to ensure NV * NS * B * H <= 65535
    B = 1  # batch size
    T = 512  # sequence length
    H = 4  # number of key/value heads
    HQ = 64  # number of query heads (GQA: HQ = H * G, G must be multiple of 16)
    K = 64  # key dimension
    V = 64  # value dimension
    block_size = 64
    block_counts = 8  # number of selected blocks
    
    return {
        'B': B,
        'T': T,
        'H': H,
        'HQ': HQ,
        'K': K,
        'V': V,
        'block_size': block_size,
        'block_counts': block_counts,
    }


def test_parallel_nsa_bwd_kernel_dkv():
    """Test function for parallel_nsa_bwd_kernel_dkv"""
    config = get_test_config()
    B, T, H, HQ, K, V = config['B'], config['T'], config['H'], config['HQ'], config['K'], config['V']
    block_size = config['block_size']
    block_counts = config['block_counts']
    
    # Load input tensors from pt file
    input_dir = "/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/parallel_nsa_bwd_kernel_dkv"
    print(f"Loading inputs from: {input_dir}/inputs.pt")
    
    device = torch.device('npu:0')
    dtype = torch.float16
    
    inputs_dict = torch.load(os.path.join(input_dir, "inputs.pt"))
    
    # Query tensor: [B, T, HQ, K]
    q = inputs_dict['q'].to(device)
    
    # Key tensor: [B, T, H, K]
    k = inputs_dict['k'].to(device)
    
    # Value tensor: [B, T, H, V]
    v = inputs_dict['v'].to(device)
    
    # Output tensor: [B, T, HQ, V]
    o = inputs_dict['o'].to(device)
    
    # LSE tensor: [B, T, HQ]
    lse = inputs_dict['lse'].to(device)
    
    # Gradient of output: [B, T, HQ, V]
    do = inputs_dict['do'].to(device)
    
    # Block mask: [B, T, H, NS]
    block_mask = inputs_dict['block_mask'].to(device)
    
    # Scale factor
    scale = inputs_dict['scale']
    block_size = inputs_dict['block_size']
    
    NS = triton.cdiv(T, block_size)
    
    print(f"Test configuration:")
    print(f"  Batch size (B): {B}")
    print(f"  Sequence length (T): {T}")
    print(f"  Key/Value heads (H): {H}")
    print(f"  Query heads (HQ): {HQ}")
    print(f"  Key dimension (K): {K}")
    print(f"  Value dimension (V): {V}")
    print(f"  Block size: {block_size}")
    print(f"  Block counts: {block_counts}")
    print(f"  Number of blocks (NS): {NS}")
    print(f"  Query shape: {q.shape}")
    print(f"  Key shape: {k.shape}")
    print(f"  Value shape: {v.shape}")
    print(f"  Output shape: {o.shape}")
    print(f"  LSE shape: {lse.shape}")
    print(f"  dO shape: {do.shape}")
    print(f"  Block mask shape: {block_mask.shape}")
    
    # Execute the kernel once
    print("\nExecuting kernel...")
    dk_npu, dv_npu = parallel_nsa_bwd_dkv(
        q=q,
        k=k,
        v=v,
        o=o,
        lse=lse,
        do=do,
        block_mask=block_mask,
        block_size=block_size,
        scale=scale,
        cu_seqlens=None,
        chunk_indices=None,
    )
    torch.npu.synchronize()
    
    # Load expected outputs and compare
    print("\nLoading expected outputs and comparing...")
    outputs_dict = torch.load(os.path.join(input_dir, "outputs.pt"))
    
    dk_gpu_fp64 = outputs_dict['dk_fp64'].to(device)
    dv_gpu_fp64 = outputs_dict['dv_fp64'].to(device)
    dk_gpu = outputs_dict['dk'].to(device)
    dv_gpu = outputs_dict['dv'].to(device)
    
    # Compare outputs with precision check
    print("\ndk comparison:")
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(dk_npu, dk_gpu_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(dk_gpu, dk_gpu_fp64)
    print(f"最大相对误差比 :{relative_max_npu / relative_max_gpu}, 平均相对误差比 :{relative_mean_npu / relative_mean_gpu}, 均方根误差比 :{rmse_npu / rmse_gpu}")
    assert relative_max_npu / relative_max_gpu <= 2, "dk 最大相对误差比 check failed"
    assert relative_mean_npu / relative_mean_gpu <= 1.2, "dk 平均相对误差比 check failed"
    assert rmse_npu / rmse_gpu <= 1.2, "dk 均方根误差比 check failed"
    
    print("\ndv comparison:")
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(dv_npu, dv_gpu_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(dv_gpu, dv_gpu_fp64)
    print(f"最大相对误差比 :{relative_max_npu / relative_max_gpu}, 平均相对误差比 :{relative_mean_npu / relative_mean_gpu}, 均方根误差比 :{rmse_npu / rmse_gpu}")
    assert relative_max_npu / relative_max_gpu <= 2, "dv 最大相对误差比 check failed"
    assert relative_mean_npu / relative_mean_gpu <= 1.2, "dv 平均相对误差比 check failed"
    assert rmse_npu / rmse_gpu <= 1.2, "dv 均方根误差比 check failed"
    
    print("✓ All outputs comparison passed!")
    
    # Warmup run
    print("\nWarming up...")
    _ = parallel_nsa_bwd_dkv(
        q=q,
        k=k,
        v=v,
        o=o,
        lse=lse,
        do=do,
        block_mask=block_mask,
        block_size=block_size,
        scale=scale,
        cu_seqlens=None,
        chunk_indices=None,
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
            _ = parallel_nsa_bwd_dkv(
                q=q,
                k=k,
                v=v,
                o=o,
                lse=lse,
                do=do,
                block_mask=block_mask,
                block_size=block_size,
                scale=scale,
                cu_seqlens=None,
                chunk_indices=None,
            )
            torch.npu.synchronize()
            prof.step()
        prof.stop()
    
    print("\nProfiling completed!")
    print("Results saved to: ./result_dir_original")
    
    print(f"\nOutput dK shape: {dk_npu.shape}")
    print(f"Output dK dtype: {dk_npu.dtype}")
    print(f"Output dV shape: {dv_npu.shape}")
    print(f"Output dV dtype: {dv_npu.dtype}")


if __name__ == "__main__":
    test_parallel_nsa_bwd_kernel_dkv()
