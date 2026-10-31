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


@triton.jit
def exp(x):
    return tl.exp(x)

# Copy kernel definition from original file
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
@triton.jit(do_not_specialize=['T', 'TC'])
def parallel_nsa_compression_bwd_kernel_dkv(
    q,
    k,
    v,
    lse,
    delta,
    do,
    dk,
    dv,
    cu_seqlens,
    chunk_indices,
    chunk_offsets,
    scale,
    T,
    TC,
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
    i_v, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    all = B * TC

    if IS_VARLEN:
        i_n, i_c = tl.load(chunk_indices + i_c * 2).to(tl.int32), tl.load(chunk_indices + i_c * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        # the number of compression representations in total
        TC = tl.cdiv(T, BS)
        boc = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_b * T, i_b * T + T
        boc = i_b * tl.cdiv(T, BS)

    p_k = tl.make_block_ptr(k + (boc * H + i_h) * K, (TC, K), (H*K, 1), (i_c * BC, 0), (BC, BK), (1, 0))
    p_v = tl.make_block_ptr(v + (boc * H + i_h) * V, (TC, V), (H*V, 1), (i_c * BC, i_v * BV), (BC, BV), (1, 0))
    p_dk = tl.make_block_ptr(dk + (i_v * all*H + boc * H + i_h) * K, (TC, K), (H*K, 1), (i_c * BC, 0), (BC, BK), (1, 0))
    p_dv = tl.make_block_ptr(dv + (boc * H + i_h) * V, (TC, V), (H*V, 1), (i_c * BC, i_v * BV), (BC, BV), (1, 0))

    # [BC, BK]
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_dk = tl.zeros([BC, BK], dtype=tl.float32)
    # [BC, BV]
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_dv = tl.zeros([BC, BV], dtype=tl.float32)

    for i in range(i_c * BC * BS, T):
        o_c = i_c * BC + tl.arange(0, BC)

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
        # [BC, G]
        b_s = tl.dot(b_k, tl.trans(b_q))
        b_p = exp(b_s - b_lse[None, :])
        b_p = tl.where((i >= max(0, (o_c + 1) * BS - 1))[:, None], b_p, 0)
        # [BC, G] @ [G, BV] -> [BC, BV]
        b_dv += tl.dot(b_p.to(b_do.dtype), b_do)
        # [BC, BV] @ [BV, G] -> [BC, G]
        b_dp = tl.dot(b_v, tl.trans(b_do))
        # [BC, G]
        b_ds = b_p * (b_dp - b_delta[None, :])
        # [BC, G] @ [G, BK] -> [BC, BK]
        b_dk += tl.dot(b_ds.to(b_q.dtype), b_q)

    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


# Copy call function from original file
def parallel_nsa_compression_bwd(
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
    B, T, HQ, K, V = *q.shape, v.shape[-1]
    TC = k.shape[1]
    H = k.shape[2]
    G = HQ // H
    BC = BS = block_size
    BK = max(triton.next_power_of_2(K), 16)
    BV = min(128, max(triton.next_power_of_2(v.shape[-1]), 16))
    NV = triton.cdiv(V, BV)
    
    NC = triton.cdiv(triton.cdiv(T, BS), BC)

    # Compute delta = sum(do * o, dim=-1)
    delta = (do * o).sum(-1)

    dk = torch.empty(NV, *k.shape, dtype=k.dtype if NV == 1 else torch.float, device=q.device)
    dv = torch.empty(v.shape, dtype=v.dtype, device=q.device)

    grid = (NV, NC, B * H)
    parallel_nsa_compression_bwd_kernel_dkv[grid](
        q=q,
        k=k,
        v=v,
        lse=lse,
        delta=delta,
        do=do,
        dk=dk,
        dv=dv,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=None,
        scale=scale,
        T=T,
        TC=TC,
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
    dk = dk.sum(0)
    return dk, dv


def test_parallel_nsa_compression_bwd_kernel_dkv():
    # Config from official test script: (B, T, H, HQ, D, S, block_size, scale, dtype) = (3, 1024, 2, 32, 60, 16, 32, 0.1, torch.float16)
    # q shape: [B, T, HQ, K] = [3, 1024, 32, 60]
    # k shape: [B, TC, H, K] = [3, 32, 2, 60] where TC = T // BS = 1024 // 32 = 32
    # v shape: [B, TC, H, V] = [3, 32, 2, 60]
    # o shape: [B, T, HQ, V] = [3, 1024, 32, 60]
    # lse shape: [B, T, HQ] = [3, 1024, 32]
    # do shape: [B, T, HQ, V] = [3, 1024, 32, 60]
    # dk shape: [B, TC, H, K] = [3, 32, 2, 60]
    # dv shape: [B, TC, H, V] = [3, 32, 2, 60]
    
    # Load inputs from saved pt file
    device = torch.device('npu:0')
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/parallel_nsa_compression_bwd_kernel_dkv'
    inputs_dict = torch.load(os.path.join(load_dir, 'inputs.pt'))
    
    # Convert inputs to npu
    q = inputs_dict['q'].to(device)
    k = inputs_dict['k'].to(device)
    v = inputs_dict['v'].to(device)
    o = inputs_dict['o'].to(device)
    lse = inputs_dict['lse'].to(device)
    do = inputs_dict['do'].to(device)
    block_size = inputs_dict['block_size']
    scale = inputs_dict['scale']
    
    print(f"Inputs loaded from {load_dir}/inputs.pt")
    print(f"q shape: {q.shape}, k shape: {k.shape}, v shape: {v.shape}, o shape: {o.shape}, lse shape: {lse.shape}, do shape: {do.shape}")
    
    # Execute kernel once for correctness check
    dk, dv = parallel_nsa_compression_bwd(
        q=q,
        k=k,
        v=v,
        o=o,
        lse=lse,
        do=do,
        block_size=block_size,
        scale=scale,
    )
    torch.npu.synchronize()
    
    # Load expected outputs and compare
    expected_outputs = torch.load(os.path.join(load_dir, 'outputs.pt'))
    expected_outputs_fp64 = torch.load(os.path.join(load_dir, 'outputs_fp64.pt'))
    
    # Convert expected outputs to npu for comparison
    expected_dk = expected_outputs['dk'].to(device)
    expected_dv = expected_outputs['dv'].to(device)
    expected_dk_fp64 = expected_outputs_fp64['dk_fp64'].to(device)
    expected_dv_fp64 = expected_outputs_fp64['dv_fp64'].to(device)
    
    # Compare outputs with new precision logic
    print("\nComparing dk outputs...")
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(dk, expected_dk_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(expected_dk, expected_dk_fp64)
    print(f"最大相对误差比 :{relative_max_npu / relative_max_gpu}, 平均相对误差比 :{relative_mean_npu / relative_mean_gpu}, 均方根误差比 :{rmse_npu / rmse_gpu}")
    print(f"relative_max_npu: {relative_max_npu}; relative_mean_npu: {relative_mean_npu}; rmse_npu: {rmse_npu}")
    print(f"relative_max_gpu: {relative_max_gpu}; relative_mean_gpu: {relative_mean_gpu}; rmse_gpu: {rmse_gpu}")
    assert relative_max_npu / relative_max_gpu <= 2, "dk 最大相对误差比 check failed"
    assert relative_mean_npu / relative_mean_gpu <= 1.2, "dk 平均相对误差比 check failed"
    assert rmse_npu / rmse_gpu <= 1.2, "dk 均方根误差比 check failed"
    
    print("\nComparing dv outputs...")
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(dv, expected_dv_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(expected_dv, expected_dv_fp64)
    print(f"最大相对误差比 :{relative_max_npu / relative_max_gpu}, 平均相对误差比 :{relative_mean_npu / relative_mean_gpu}, 均方根误差比 :{rmse_npu / rmse_gpu}")
    assert relative_max_npu / relative_max_gpu <= 2, "dv 最大相对误差比 check failed"
    assert relative_mean_npu / relative_mean_gpu <= 1.2, "dv 平均相对误差比 check failed"
    assert rmse_npu / rmse_gpu <= 1.2, "dv 均方根误差比 check failed"
    
    print("All outputs match within tolerance!")
    
    # Warmup run
    dk, dv = parallel_nsa_compression_bwd(
        q=q,
        k=k,
        v=v,
        o=o,
        lse=lse,
        do=do,
        block_size=block_size,
        scale=scale,
    )
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
            dk, dv = parallel_nsa_compression_bwd(
                q=q,
                k=k,
                v=v,
                o=o,
                lse=lse,
                do=do,
                block_size=block_size,
                scale=scale,
            )
            torch.npu.synchronize()
            prof.step()
        prof.stop()
    
    print(f"Test passed! dk shape: {dk.shape}, dv shape: {dv.shape}")
    print(f"Profiling results saved to ./result_dir_original")


if __name__ == "__main__":
    test_parallel_nsa_compression_bwd_kernel_dkv()
