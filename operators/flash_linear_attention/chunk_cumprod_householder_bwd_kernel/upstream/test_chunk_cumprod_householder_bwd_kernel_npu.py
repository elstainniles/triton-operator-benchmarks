# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl
import torch_npu
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


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit()
def chunk_cumprod_householder_bwd_kernel(
    hc_suffix, dhc_whole,
    k, dk, w1, w2, dw1, dw2, dk_new,
    cu_seqlens, split_indices, chunk_offsets, split_offsets,
    BT: tl.constexpr,  # previous small chunk size
    K: tl.constexpr,
    BK: tl.constexpr,
    T: tl.constexpr,
    S: tl.constexpr,
    G: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_ss, i_hq = tl.program_id(0), tl.program_id(1)
    i_h = i_hq // G

    if IS_VARLEN:
        i_n, i_s = tl.load(split_indices + i_ss * 2).to(tl.int32), tl.load(split_indices + i_ss * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NS = tl.cdiv(T, S)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
        boh_large = tl.load(split_offsets + i_n).to(tl.int32)
    else:
        NS = tl.cdiv(T, S)
        i_n, i_s = i_ss // NS, i_ss % NS
        bos, eos = i_n * T, i_n * T + T
        boh = i_n * tl.cdiv(T, BT)
        boh_large = i_n * tl.cdiv(T, S)

    # offset calculations
    dhc_whole += ((boh_large + i_s) * HQ + i_hq) * K * K
    hc_suffix += ((boh + tl.cdiv(i_s * S, BT)) * H + i_h) * K * K
    k += (bos * H + i_h) * K
    w1 += (bos * H + i_h) * K
    w2 += (bos * H + i_h) * K
    dw1 += (bos * HQ + i_hq) * K
    dw2 += (bos * HQ + i_hq) * K

    # dh += ((boh + tl.cdiv(i_s * S, BT)) * HQ + i_hq) * K * K
    dk += (bos * HQ + i_hq) * K
    dk_new += (bos * HQ + i_hq) * K

    stride_h = H * K * K
    NT_small = tl.cdiv(min(S, T-i_s*S), BT)
    p_dhc_whole = tl.make_block_ptr(dhc_whole, (K, K), (K, 1), (0, 0), (BK, BK), (1, 0))
    b_dhc = tl.zeros([BK, BK], dtype=tl.float32)
    b_dhc += tl.load(p_dhc_whole, boundary_check=(0, 1))

    # calculate dh
    for i_t_small in range(0, NT_small):
        p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_s*S + i_t_small*BT, 0), (BT, BK), (1, 0))
        p_dk = tl.make_block_ptr(dk, (T, K), (HQ*K, 1), (i_s*S + i_t_small*BT, 0), (BT, BK), (1, 0))
        p_dk_new = tl.make_block_ptr(dk_new, (T, K), (HQ*K, 1), (i_s*S + i_t_small*BT, 0), (BT, BK), (1, 0))
        p_hc = tl.make_block_ptr(hc_suffix + i_t_small * stride_h, (K, K), (K, 1), (0, 0), (BK, BK), (1, 0))

        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_dk = tl.load(p_dk, boundary_check=(0, 1))

        p_w1 = tl.make_block_ptr(w1, (T, K), (H*K, 1), (i_s*S + i_t_small*BT, 0), (BT, BK), (1, 0))
        p_w2 = tl.make_block_ptr(w2, (T, K), (H*K, 1), (i_s*S + i_t_small*BT, 0), (BT, BK), (1, 0))

        b_w1 = tl.load(p_w1, boundary_check=(0, 1))
        b_w2 = tl.load(p_w2, boundary_check=(0, 1))
        b_hc = tl.load(p_hc, boundary_check=(0, 1))

        b_dk_new = b_dk - tl.dot(b_dk.to(b_hc.dtype), b_hc)
        tl.store(p_dk_new, b_dk_new.to(dk_new.dtype.element_ty), boundary_check=(0, 1))

        b_dh = b_dhc - tl.dot(tl.trans(b_hc), b_dhc.to(b_hc.dtype))
        b_dw2 = tl.dot(b_w1, b_dh.to(b_w1.dtype))
        b_dw1 = tl.dot(b_w2, tl.trans(b_dh.to(b_w2.dtype)))

        p_dw1 = tl.make_block_ptr(dw1, (T, K), (HQ*K, 1), (i_s*S + i_t_small*BT, 0), (BT, BK), (1, 0))
        p_dw2 = tl.make_block_ptr(dw2, (T, K), (HQ*K, 1), (i_s*S + i_t_small*BT, 0), (BT, BK), (1, 0))

        tl.store(p_dw1, b_dw1.to(dw1.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dw2, b_dw2.to(dw2.dtype.element_ty), boundary_check=(0, 1))

        b_dhc = b_dhc - tl.dot(tl.dot(b_dhc.to(b_w2.dtype), tl.trans(b_w2)).to(b_w1.dtype), b_w1)
        b_dhc -= tl.dot(tl.trans(b_dk).to(b_k.dtype), b_k)


def chunk_cumprod_householder_bwd_fn(
    w1: torch.Tensor,
    w2: torch.Tensor,
    hc_suffix: torch.Tensor,
    dhc_whole: torch.Tensor,
    k: torch.Tensor,
    dk: torch.Tensor,
    S: int,  # split size, aka large chunk size
    BT: int,  # small chunk size
    cu_seqlens: torch.Tensor = None,
    chunk_indices: torch.LongTensor | None = None,
):
    B, T, HQ, K = dk.shape
    H = k.shape[2]
    G = HQ // H

    # Simplified version for testing - in real code this uses prepare_chunk_indices
    if chunk_indices is None and cu_seqlens is not None:
        # Would call prepare_chunk_indices(cu_seqlens, S)
        chunk_indices = None
    split_indices = chunk_indices
    
    # Simplified version for testing - in real code this uses prepare_chunk_offsets
    chunk_offsets = None if cu_seqlens is None else None  # Would call prepare_chunk_offsets(cu_seqlens, BT)
    split_offsets = None if cu_seqlens is None else None  # Would call prepare_chunk_offsets(cu_seqlens, S)

    if cu_seqlens is None:
        N = B
        NS = N * triton.cdiv(T, S)
    else:
        N = len(cu_seqlens) - 1
        NS = split_offsets[-1].item()

    grid = (NS, HQ)
    dw1 = torch.empty_like(dk, dtype=torch.float32)
    dw2 = torch.empty_like(dk, dtype=torch.float32)
    dk_new = torch.empty_like(dk, dtype=torch.float32)

    # Simplified check_shared_mem for testing
    check_shared_mem_ampere = True  # Assume ampere or better for testing

    chunk_cumprod_householder_bwd_kernel[grid](
        hc_suffix=hc_suffix, dhc_whole=dhc_whole,
        k=k, dk=dk, w1=w1, w2=w2, dw1=dw1, dw2=dw2, dk_new=dk_new,
        cu_seqlens=cu_seqlens,
        split_indices=split_indices, chunk_offsets=chunk_offsets, split_offsets=split_offsets,
        BT=BT, K=K, G=G, H=H, HQ=HQ, BK=K,
        T=T, S=S,
        num_warps=8 if K == 128 else 4,
        num_stages=2 if check_shared_mem_ampere else 1,
    )
    return dw1, dw2, dk_new


def test_chunk_cumprod_householder_bwd_kernel():
    """Test function for chunk_cumprod_householder_bwd_kernel with profiling"""
    
    # Load input tensors from pt files
    load_dir = "/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_cumprod_householder_bwd_kernel"
    inputs_dict = torch.load(os.path.join(load_dir, "inputs.pt"))
    
    device = torch.device('npu')
    
    # Convert loaded tensors to NPU
    w1 = inputs_dict['w1'].to(device)
    w2 = inputs_dict['w2'].to(device)
    k = inputs_dict['k'].to(device)
    dk = inputs_dict['dk'].to(device)
    hc_suffix = inputs_dict['hc_suffix'].to(device)
    dhc_whole = inputs_dict['dhc_whole'].to(device)
    cu_seqlens = inputs_dict['cu_seqlens']
    S = inputs_dict['S']
    BT = inputs_dict['BT']
    
    # Calculate derived shapes
    B, T, HQ, K = dk.shape
    H = k.shape[2]
    num_chunks = B * triton.cdiv(T, BT)
    num_splits = B * triton.cdiv(T, S)
    
    print(f"Testing chunk_cumprod_householder_bwd_kernel with config:")
    print(f"  B={B}, T={T}, HQ={HQ}, H={H}, K={K}")
    print(f"  S={S}, BT={BT}")
    print(f"  num_chunks={num_chunks}, num_splits={num_splits}")
    print(f"  Device: {device}")
    
    # Execute the kernel once
    print("\nExecuting kernel...")
    dw1_npu, dw2_npu, dk_new_npu = chunk_cumprod_householder_bwd_fn(
        w1=w1,
        w2=w2,
        hc_suffix=hc_suffix,
        dhc_whole=dhc_whole,
        k=k,
        dk=dk,
        S=S,
        BT=BT,
        cu_seqlens=cu_seqlens,
    )
    torch.npu.synchronize()
    
    # Load expected output tensors from pt file
    outputs_dict = torch.load(os.path.join(load_dir, "outputs.pt"))
    dw1_gpu_fp64 = outputs_dict['dw1_fp64'].to(device)
    dw2_gpu_fp64 = outputs_dict['dw2_fp64'].to(device)
    dk_new_gpu_fp64 = outputs_dict['dk_new_fp64'].to(device)
    dw1_gpu = outputs_dict['dw1'].to(device)
    dw2_gpu = outputs_dict['dw2'].to(device)
    dk_new_gpu = outputs_dict['dk_new'].to(device)
    
    # Compare outputs with expected values
    print("\nComparing dw1 outputs with expected values...")
    relative_max_npu_dw1, relative_mean_npu_dw1, rmse_npu_dw1 = cal_precision(dw1_npu, dw1_gpu_fp64)
    relative_max_gpu_dw1, relative_mean_gpu_dw1, rmse_gpu_dw1 = cal_precision(dw1_gpu, dw1_gpu_fp64)
    print(f"dw1 最大相对误差比 :{relative_max_npu_dw1 / relative_max_gpu_dw1}, 平均相对误差比 :{relative_mean_npu_dw1 / relative_mean_gpu_dw1}, 均方根误差比 :{rmse_npu_dw1 / rmse_gpu_dw1}")
    assert relative_max_npu_dw1 / relative_max_gpu_dw1 <= 2, "dw1 最大相对误差比 check failed"
    assert relative_mean_npu_dw1 / relative_mean_gpu_dw1 <= 1.2, "dw1 平均相对误差比 check failed"
    assert rmse_npu_dw1 / rmse_gpu_dw1 <= 1.2, "dw1 均方根误差比 check failed"
    
    print("\nComparing dw2 outputs with expected values...")
    relative_max_npu_dw2, relative_mean_npu_dw2, rmse_npu_dw2 = cal_precision(dw2_npu, dw2_gpu_fp64)
    relative_max_gpu_dw2, relative_mean_gpu_dw2, rmse_gpu_dw2 = cal_precision(dw2_gpu, dw2_gpu_fp64)
    print(f"dw2 最大相对误差比 :{relative_max_npu_dw2 / relative_max_gpu_dw2}, 平均相对误差比 :{relative_mean_npu_dw2 / relative_mean_gpu_dw2}, 均方根误差比 :{rmse_npu_dw2 / rmse_gpu_dw2}")
    assert relative_max_npu_dw2 / relative_max_gpu_dw2 <= 2, "dw2 最大相对误差比 check failed"
    assert relative_mean_npu_dw2 / relative_mean_gpu_dw2 <= 1.2, "dw2 平均相对误差比 check failed"
    assert rmse_npu_dw2 / rmse_gpu_dw2 <= 1.2, "dw2 均方根误差比 check failed"
    
    print("\nComparing dk_new outputs with expected values...")
    relative_max_npu_dk, relative_mean_npu_dk, rmse_npu_dk = cal_precision(dk_new_npu, dk_new_gpu_fp64)
    relative_max_gpu_dk, relative_mean_gpu_dk, rmse_gpu_dk = cal_precision(dk_new_gpu, dk_new_gpu_fp64)
    print(f"dk_new 最大相对误差比 :{relative_max_npu_dk / relative_max_gpu_dk}, 平均相对误差比 :{relative_mean_npu_dk / relative_mean_gpu_dk}, 均方根误差比 :{rmse_npu_dk / rmse_gpu_dk}")
    assert relative_max_npu_dk / relative_max_gpu_dk <= 2, "dk_new 最大相对误差比 check failed"
    assert relative_mean_npu_dk / relative_mean_gpu_dk <= 1.2, "dk_new 平均相对误差比 check failed"
    assert rmse_npu_dk / rmse_gpu_dk <= 1.2, "dk_new 均方根误差比 check failed"
    print("Output comparison passed!")
    
    # Warmup run
    print("\nWarmup run...")
    _ = chunk_cumprod_householder_bwd_fn(
        w1=w1,
        w2=w2,
        hc_suffix=hc_suffix,
        dhc_whole=dhc_whole,
        k=k,
        dk=dk,
        S=S,
        BT=BT,
        cu_seqlens=cu_seqlens,
    )
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
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result_dir_original")
    ) as prof:
        prof.start()
        for i in range(35):
            _ = chunk_cumprod_householder_bwd_fn(
                w1=w1,
                w2=w2,
                hc_suffix=hc_suffix,
                dhc_whole=dhc_whole,
                k=k,
                dk=dk,
                S=S,
                BT=BT,
                cu_seqlens=cu_seqlens,
            )
            torch.npu.synchronize()
            prof.step()
        prof.stop()
    
    print("\nProfiling completed.")
    print(f"Profiling results saved to: ./result_dir_original")
    
    print("\nTest passed!")
    print(f"Output shapes: dw1={dw1_npu.shape}, dw2={dw2_npu.shape}, dk_new={dk_new_npu.shape}")


if __name__ == "__main__":
    test_chunk_cumprod_householder_bwd_kernel()
