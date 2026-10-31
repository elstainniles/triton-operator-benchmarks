# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import torch_npu
import triton
import triton.language as tl


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
    'USE_GATE': lambda args: args['g_cumsum'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def parallel_path_fwd_kernel(
    q,
    k,
    v,
    o,
    o_new,
    g_cumsum,
    w1,
    w2,
    scale,
    L,
    L_new,
    M,
    cu_seqlens,
    indices,
    T,
    G: tl.constexpr,
    HQ: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_GATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    if IS_VARLEN:
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        i_n = i_b
        bos, eos = i_n * T, i_n * T + T

    p_q = tl.make_block_ptr(q + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    b_q = tl.zeros([BT, BK], dtype=tl.float32)
    b_q += tl.load(p_q, boundary_check=(0, 1))
    sm_scale = scale * 1.44269504
    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    p_o = tl.make_block_ptr(o + (bos * HQ + i_hq) * V, (T, V), (HQ*V, 1), (i_t * BT, 0), (BT, BV), (1, 0))
    b_o += tl.load(p_o, boundary_check=(0, 1))

    p_L = tl.make_block_ptr(L + bos * HQ + i_hq, (T, ), (HQ, ), (i_t * BT, ), (BT, ), (0,))
    p_M = tl.make_block_ptr(M + bos * HQ + i_hq, (T, ), (HQ, ), (i_t * BT, ), (BT, ), (0,))
    b_l = tl.load(p_L, boundary_check=(0,))
    b_m = tl.load(p_M, boundary_check=(0,))

    if USE_GATE:
        p_g_cumsum_q = tl.make_block_ptr(g_cumsum + bos * HQ + i_hq, (T, ), (HQ, ), (i_t * BT, ), (BT, ), (0,))
        b_g_cumsum_q = tl.load(p_g_cumsum_q, boundary_check=(0,))
    else:
        b_g_cumsum_q = None

    for offset in range((i_t + 1) * BT - 2 * BS, i_t*BT-BS, -BS):
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (K, T), (1, K*H), (0, offset), (BK, BS), (0, 1))  # GQA when H!=HQ
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (V*H, 1), (offset, 0), (BS, BV), (1, 0))  # GQA when H!=HQ
        p_w1 = tl.make_block_ptr(w1 + (bos * H + i_h) * K, (K, T), (1, K*H), (0, offset), (BK, BS), (0, 1))
        p_w2 = tl.make_block_ptr(w2 + (bos * H + i_h) * K, (T, K), (K*H, 1), (offset, 0), (BS, BK), (1, 0))
        # [BK, BS]

        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BS, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BK, BK]
        b_w1 = tl.load(p_w1, boundary_check=(0, 1))
        b_w2 = tl.load(p_w2, boundary_check=(0, 1))
        # [BT, BS]
        m_s = i_t * BT + tl.arange(0, BT) >= (offset + BS)
        b_s = tl.dot(b_q.to(b_k.dtype), b_k)

        if USE_GATE:
            p_g_cumsum_k = tl.make_block_ptr(g_cumsum + (bos * HQ + i_hq), (T, ), (HQ, ), (offset, ), (BS, ), (0,))
            b_g_cumsum_k = tl.load(p_g_cumsum_k, boundary_check=(0,))
            b_s = b_s + b_g_cumsum_q[:, None] - b_g_cumsum_k[None, :]
        b_s = tl.where(m_s[:, None], b_s * sm_scale, float("-inf"))
        b_m_new = tl.maximum(b_m, tl.max(b_s, 1))
        alpha = tl.math.exp2(b_m - b_m_new)
        b_s = tl.math.exp2(b_s - b_m_new[:, None])
        b_o *= alpha[:, None]
        b_l = b_l * alpha + tl.sum(b_s, 1)
        b_m = b_m_new
        b_o += tl.dot(b_s.to(b_v.dtype), b_v)
        b_s2 = tl.dot(b_q.to(b_w1.dtype), b_w1)
        b_s2 = tl.where(m_s[:, None], b_s2, 0)
        b_q -= tl.dot(b_s2.to(b_w2.dtype), b_w2)
        # tl.store("b_q: {}, offset: {} ", b_q, offset)

    tl.debug_barrier()

    for offset in range(i_t * BT - BS, -BS, -BS):
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (K, T), (1, K*H), (0, offset), (BK, BS), (0, 1))  # GQA when H!=HQ
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (V*H, 1), (offset, 0), (BS, BV), (1, 0))  # GQA when H!=HQ
        p_w1 = tl.make_block_ptr(w1 + (bos * H + i_h) * K, (K, T), (1, K*H), (0, offset), (BK, BS), (0, 1))
        p_w2 = tl.make_block_ptr(w2 + (bos * H + i_h) * K, (T, K), (K*H, 1), (offset, 0), (BS, BK), (1, 0))
        # [BK, BS]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BS, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_w1 = tl.load(p_w1, boundary_check=(0, 1))
        b_w2 = tl.load(p_w2, boundary_check=(0, 1))
        # [BT, BS]
        b_s = tl.dot(b_q.to(b_k.dtype), b_k)
        if USE_GATE:
            p_g_cumsum_k = tl.make_block_ptr(g_cumsum + (bos * HQ + i_hq), (T, ), (HQ, ), (offset, ), (BS, ), (0,))
            b_g_cumsum_k = tl.load(p_g_cumsum_k, boundary_check=(0,))
            b_s = b_s + b_g_cumsum_q[:, None] - b_g_cumsum_k[None, :]
        b_s = b_s * sm_scale
        b_m_new = tl.maximum(b_m, tl.max(b_s, 1))
        alpha = tl.math.exp2(b_m - b_m_new)
        b_s = tl.math.exp2(b_s - b_m_new[:, None])
        b_o *= alpha[:, None]
        b_l = b_l * alpha + tl.sum(b_s, 1)
        b_m = b_m_new
        b_o += tl.dot(b_s.to(b_v.dtype), b_v)
        b_s2 = tl.dot(b_q.to(b_w1.dtype), b_w1)
        b_q -= tl.dot(b_s2.to(b_w2.dtype), b_w2)

    b_o = b_o / b_l[:, None]
    p_o_new = tl.make_block_ptr(o_new + (bos * HQ + i_hq) * V, (T, V), (HQ*V, 1), (i_t*BT, 0), (BT, BV), (1, 0))
    tl.store(p_o_new, b_o.to(p_o_new.dtype.element_ty), boundary_check=(0, 1))
    b_l = tl.math.log2(b_l) + b_m
    p_L_new = tl.make_block_ptr(L_new + (bos * HQ + i_hq), (T, ), (HQ, ), (i_t * BT, ), (BT, ), (0,))
    tl.store(p_L_new, b_l.to(p_L_new.dtype.element_ty), boundary_check=(0,))


def parallel_path_fwd_fn(
    q,
    k,
    v,
    o,
    g_cumsum,
    w1,
    w2,
    scale,
    L,
    M,
    cu_seqlens,
    BT,
    BS,
    chunk_indices: torch.LongTensor | None = None,
):
    B, T, HQ, K = q.shape
    V = v.shape[-1]
    H = k.shape[-2]
    G = HQ // H
    
    # Prepare chunk indices if needed
    if chunk_indices is None and cu_seqlens is not None:
        # Simplified version for testing - in real code this uses prepare_chunk_indices
        chunk_indices = None
    
    indices = chunk_indices
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(indices)
    grid = (NT, B * HQ)
    o_new = torch.empty_like(o, dtype=v.dtype)
    L_new = torch.empty_like(L)

    parallel_path_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        o=o,
        o_new=o_new,
        w1=w1,
        w2=w2,
        g_cumsum=g_cumsum,
        scale=scale,
        cu_seqlens=cu_seqlens,
        indices=indices,
        L=L,
        L_new=L_new,
        M=M,
        T=T,
        K=K,
        V=V,
        BK=triton.next_power_of_2(K),
        BV=triton.next_power_of_2(V),
        G=G,
        HQ=HQ,
        H=H,
        BS=BS,
        BT=BT,
        num_warps=8 if (BT == 128 and K == 128) else 4,
    )
    return o_new, L_new


def test_parallel_path_fwd_kernel():
    """Test function for parallel_path_fwd_kernel with profiling"""
    
    # Load input tensors from pt file
    import os
    load_dir = "/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/parallel_path_fwd_kernel_bf16"
    inputs_dict = torch.load(os.path.join(load_dir, "inputs.pt"))
    
    device = torch.device('npu')
    
    # Convert loaded tensors to NPU
    q = inputs_dict['q'].to(device)
    k = inputs_dict['k'].to(device)
    v = inputs_dict['v'].to(device)
    o = inputs_dict['o'].to(device)
    w1 = inputs_dict['w1'].to(device)
    w2 = inputs_dict['w2'].to(device)
    g_cumsum = inputs_dict['g_cumsum']
    scale = inputs_dict['scale']
    L = inputs_dict['L'].to(device)
    M = inputs_dict['M'].to(device)
    cu_seqlens = inputs_dict['cu_seqlens']
    BT = inputs_dict['BT']
    BS = inputs_dict['BS']
    
    print(f"Testing parallel_path_fwd_kernel with config:")
    print(f"  q shape: {q.shape}, k shape: {k.shape}, v shape: {v.shape}")
    print(f"  BT={BT}, BS={BS}")
    print(f"  Device: {device}")
    
    # Execute kernel once
    print("\nExecuting kernel...")
    o_new_npu, L_new_npu = parallel_path_fwd_fn(
        q=q,
        k=k,
        v=v,
        o=o,
        g_cumsum=g_cumsum,
        w1=w1,
        w2=w2,
        scale=scale,
        L=L,
        M=M,
        cu_seqlens=cu_seqlens,
        BT=BT,
        BS=BS,
    )
    
    # Load expected output tensors from pt file
    outputs_dict = torch.load(os.path.join(load_dir, "outputs.pt"))
    o_new_gpu_fp32 = outputs_dict['o_new_fp32'].to(device)
    L_new_gpu_fp32 = outputs_dict['L_new_fp32'].to(device)
    o_new_gpu = outputs_dict['o_new'].to(device)
    L_new_gpu = outputs_dict['L_new'].to(device)
    
    # Compare outputs with expected values
    print("\nComparing o_new outputs with expected values...")
    relative_max_npu_o, relative_mean_npu_o, rmse_npu_o = cal_precision(o_new_npu, o_new_gpu_fp32)
    relative_max_gpu_o, relative_mean_gpu_o, rmse_gpu_o = cal_precision(o_new_gpu, o_new_gpu_fp32)
    print(f"o_new 最大相对误差比 :{relative_max_npu_o / relative_max_gpu_o}, 平均相对误差比 :{relative_mean_npu_o / relative_mean_gpu_o}, 均方根误差比 :{rmse_npu_o / rmse_gpu_o}")
    assert relative_max_npu_o / relative_max_gpu_o <= 2, "o_new 最大相对误差比 check failed"
    assert relative_mean_npu_o / relative_mean_gpu_o <= 1.2, "o_new 平均相对误差比 check failed"
    assert rmse_npu_o / rmse_gpu_o <= 1.2, "o_new 均方根误差比 check failed"
    
    print("\nComparing L_new outputs with expected values...")
    relative_max_npu_L, relative_mean_npu_L, rmse_npu_L = cal_precision(L_new_npu, L_new_gpu_fp32)
    relative_max_gpu_L, relative_mean_gpu_L, rmse_gpu_L = cal_precision(L_new_gpu, L_new_gpu_fp32)
    print(f"L_new 最大相对误差比 :{relative_max_npu_L / relative_max_gpu_L}, 平均相对误差比 :{relative_mean_npu_L / relative_mean_gpu_L}, 均方根误差比 :{rmse_npu_L / rmse_gpu_L}")
    assert relative_max_npu_L / relative_max_gpu_L <= 2, "L_new 最大相对误差比 check failed"
    assert relative_mean_npu_L / relative_mean_gpu_L <= 1.2, "L_new 平均相对误差比 check failed"
    assert rmse_npu_L / rmse_gpu_L <= 1.2, "L_new 均方根误差比 check failed"
    print("Output comparison passed!")
    
    # Warmup run for profiling
    print("\nWarmup run...")
    _ = parallel_path_fwd_fn(
        q=q,
        k=k,
        v=v,
        o=o,
        g_cumsum=g_cumsum,
        w1=w1,
        w2=w2,
        scale=scale,
        L=L,
        M=M,
        cu_seqlens=cu_seqlens,
        BT=BT,
        BS=BS,
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
            _ = parallel_path_fwd_fn(
                q=q,
                k=k,
                v=v,
                o=o,
                g_cumsum=g_cumsum,
                w1=w1,
                w2=w2,
                scale=scale,
                L=L,
                M=M,
                cu_seqlens=cu_seqlens,
                BT=BT,
                BS=BS,
            )
            torch.npu.synchronize()
            prof.step()
        prof.stop()
    
    print("\nProfiling completed.")
    print(f"Profiling results saved to: ./result_dir_original")
    
    print("\nTest passed!")
    print(f"Output shapes: o_new={o_new_npu.shape}, L_new={L_new_npu.shape}")


if __name__ == "__main__":
    test_parallel_path_fwd_kernel()
