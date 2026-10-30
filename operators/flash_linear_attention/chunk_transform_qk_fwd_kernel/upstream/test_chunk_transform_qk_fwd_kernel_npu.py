import torch
import torch_npu
import triton
import triton.language as tl
import os


# Simplified implementation of fwd_prepare_T (since it doesn't exist in original code)


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


def fwd_prepare_T(k: torch.Tensor, beta: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """
    Simplified version of fwd_prepare_T that computes the transition matrix A.
    This is a placeholder implementation since the original function doesn't exist.
    """
    B, H, T, K = k.shape
    BT = chunk_size
    NT = triton.cdiv(T, BT)
    
    # Create A matrix (simplified version)
    A = torch.zeros(B, H, T, BT, device=k.device, dtype=k.dtype)
    
    for i in range(NT):
        start = i * BT
        end = min((i + 1) * BT, T)
        actual_BT = end - start
        
        # Get chunk of k and beta
        k_chunk = k[:, :, start:end, :]  # [B, H, actual_BT, K]
        beta_chunk = beta[:, :, start:end]  # [B, H, actual_BT]
        
        # Compute K @ K^T
        kkt = torch.matmul(k_chunk, k_chunk.transpose(-1, -2))  # [B, H, actual_BT, actual_BT]
        
        # Apply beta
        kkt = kkt * beta_chunk.unsqueeze(-1)  # [B, H, actual_BT, actual_BT]
        
        # Apply lower triangular mask
        mask = torch.tril(torch.ones(actual_BT, actual_BT, device=k.device, dtype=torch.bool), diagonal=-1)
        kkt = kkt * mask
        
        # Store in A
        A[:, :, start:end, :actual_BT] = kkt
    
    return A


# Copy kernel definition from original file
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4]
    ],
    key=['BT', 'K', 'V'],
)
@triton.jit(do_not_specialize=['T'])
def chunk_transform_qk_fwd_kernel(
    q,
    k,
    v,
    beta,
    o,
    A,
    q_new,
    k_new,
    A_local,
    scale,
    T,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BT: tl.constexpr,
    OUTPUT_ATTENTIONS: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)

    p_q = tl.make_block_ptr(q + i_bh * T*K, (T, K), (K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * T*K, (T, K), (K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * T*V, (T, V), (V, 1), (i_t * BT, 0), (BT, BV), (1, 0))
    b_q = (tl.load(p_q, boundary_check=(0, 1)) * scale).to(p_q.dtype.element_ty)
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_v = tl.load(p_v, boundary_check=(0, 1))

    p_T = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_T = tl.load(p_T, boundary_check=(0, 1))

    o_i = tl.arange(0, BT)
    m_t = o_i[:, None] >= o_i[None, :]
    b_qk = tl.where(m_t, tl.dot(b_q, tl.trans(b_k), allow_tf32=False), 0).to(b_q.dtype)
    m_t = o_i[:, None] > o_i[None, :]
    b_kk = tl.where(m_t, tl.dot(b_k, tl.trans(b_k), allow_tf32=False), 0).to(b_k.dtype)

    p_beta = tl.make_block_ptr(beta + i_bh * T, (T, ), (1, ), (i_t * BT, ), (BT, ), (0, ))
    b_beta = tl.load(p_beta, boundary_check=(0, ))
    b_k_beta = (b_k * b_beta[:, None]).to(b_k.dtype)

    b_qkT = tl.dot(b_qk, b_T, allow_tf32=False).to(b_k.dtype)

    if OUTPUT_ATTENTIONS:
        p_a = tl.make_block_ptr(A_local + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
        tl.store(p_a, b_qkT.to(p_a.dtype.element_ty), boundary_check=(0, 1))

    b_kkT = tl.dot(b_kk, b_T, allow_tf32=False).to(b_k.dtype)
    p_o = tl.make_block_ptr(o + i_bh * T*V, (T, V), (V, 1), (i_t * BT, 0), (BT, BV), (1, 0))
    tl.store(p_o, tl.dot(b_qkT, b_v).to(p_o.dtype.element_ty), boundary_check=(0, 1))

    p_q_new = tl.make_block_ptr(q_new + i_bh * T*K, (T, K), (K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    tl.store(p_q_new, (b_q - tl.dot(b_qkT, b_k_beta, allow_tf32=False)).to(p_q_new.dtype.element_ty), boundary_check=(0, 1))

    p_k_new = tl.make_block_ptr(k_new + i_bh * T*K, (T, K), (K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    b_k_new = b_k - tl.dot(tl.trans(b_kkT), b_k_beta, allow_tf32=False)
    tl.store(p_k_new, b_k_new.to(p_k_new.dtype.element_ty), boundary_check=(0, 1))


# Copy call function from original file
def chunk_transform_qk_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    scale: float,
    chunk_size: int,
    output_attentions: bool,
):
    B, H, T, K = k.shape
    BT = chunk_size
    q_new = torch.empty_like(q)
    k_new = torch.empty_like(k)
    o = torch.empty_like(v)
    grid = (triton.cdiv(T, BT), B*H)
    V = v.shape[-1]
    A_local = torch.empty_like(A) if output_attentions else None
    chunk_transform_qk_fwd_kernel[grid](
        q,
        k,
        v,
        beta,
        o,
        A,
        q_new,
        k_new,
        A_local,
        scale=scale,
        T=T,
        K=K,
        V=V,
        BT=BT,
        BK=triton.next_power_of_2(K),
        BV=triton.next_power_of_2(V),
        OUTPUT_ATTENTIONS=output_attentions,
    )
    return q_new, k_new, o, A_local


def test_chunk_transform_qk_fwd_kernel():
    # Config with large shape for performance testing
    B, H, T, K, V = 4, 8, 2048, 128, 128
    
    # Load inputs from saved pt file
    device = torch.device('npu:0')
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_transform_qk_fwd_kernel'
    inputs_dict = torch.load(os.path.join(load_dir, 'inputs.pt'))
    
    # Convert inputs to npu
    q = inputs_dict['q'].to(device)
    k = inputs_dict['k'].to(device)
    v = inputs_dict['v'].to(device)
    beta = inputs_dict['beta'].to(device)
    A = inputs_dict['A'].to(device)
    scale = inputs_dict['scale']
    chunk_size = inputs_dict['chunk_size']
    output_attentions = inputs_dict['output_attentions']
    
    print(f"Inputs loaded from {load_dir}/inputs.pt")
    
    # Execute kernel once for correctness check
    q_new, k_new, o, A_local = chunk_transform_qk_fwd(
        q=q,
        k=k,
        v=v,
        beta=beta,
        A=A,
        scale=scale,
        chunk_size=chunk_size,
        output_attentions=output_attentions,
    )
    torch.npu.synchronize()
    
    # Load expected outputs and compare
    expected_outputs = torch.load(os.path.join(load_dir, 'outputs.pt'))
    expected_outputs_fp64 = torch.load(os.path.join(load_dir, 'outputs_fp64.pt'))
    
    # Convert expected outputs to npu for comparison
    expected_q_new = expected_outputs['q_new'].to(device)
    expected_k_new = expected_outputs['k_new'].to(device)
    expected_o = expected_outputs['o'].to(device)
    expected_A_local = expected_outputs['A_local'].to(device) if expected_outputs['A_local'] is not None else None
    
    expected_q_new_fp64 = expected_outputs_fp64['q_new_fp64'].to(device)
    expected_k_new_fp64 = expected_outputs_fp64['k_new_fp64'].to(device)
    expected_o_fp64 = expected_outputs_fp64['o_fp64'].to(device)
    expected_A_local_fp64 = expected_outputs_fp64['A_local_fp64'].to(device) if expected_outputs_fp64['A_local_fp64'] is not None else None
    
    # Compare outputs with new precision logic
    print("\nComparing q_new outputs...")
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(q_new, expected_q_new_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(expected_q_new, expected_q_new_fp64)
    print(f"最大相对误差比 :{relative_max_npu / relative_max_gpu}, 平均相对误差比 :{relative_mean_npu / relative_mean_gpu}, 均方根误差比 :{rmse_npu / rmse_gpu}")
    assert relative_max_npu / relative_max_gpu <= 2, "q_new 最大相对误差比 check failed"
    assert relative_mean_npu / relative_mean_gpu <= 1.2, "q_new 平均相对误差比 check failed"
    assert rmse_npu / rmse_gpu <= 1.2, "q_new 均方根误差比 check failed"
    
    print("\nComparing k_new outputs...")
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(k_new, expected_k_new_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(expected_k_new, expected_k_new_fp64)
    print(f"最大相对误差比 :{relative_max_npu / relative_max_gpu}, 平均相对误差比 :{relative_mean_npu / relative_mean_gpu}, 均方根误差比 :{rmse_npu / rmse_gpu}")
    assert relative_max_npu / relative_max_gpu <= 2, "k_new 最大相对误差比 check failed"
    assert relative_mean_npu / relative_mean_gpu <= 1.2, "k_new 平均相对误差比 check failed"
    assert rmse_npu / rmse_gpu <= 1.2, "k_new 均方根误差比 check failed"
    
    print("\nComparing o outputs...")
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(o, expected_o_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(expected_o, expected_o_fp64)
    print(f"最大相对误差比 :{relative_max_npu / relative_max_gpu}, 平均相对误差比 :{relative_mean_npu / relative_mean_gpu}, 均方根误差比 :{rmse_npu / rmse_gpu}")
    assert relative_max_npu / relative_max_gpu <= 2, "o 最大相对误差比 check failed"
    assert relative_mean_npu / relative_mean_gpu <= 1.2, "o 平均相对误差比 check failed"
    assert rmse_npu / rmse_gpu <= 1.2, "o 均方根误差比 check failed"
    
    if expected_A_local is not None and expected_A_local_fp64 is not None:
        print("\nComparing A_local outputs...")
        relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(A_local, expected_A_local_fp64)
        relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(expected_A_local, expected_A_local_fp64)
        print(f"最大相对误差比 :{relative_max_npu / relative_max_gpu}, 平均相对误差比 :{relative_mean_npu / relative_mean_gpu}, 均方根误差比 :{rmse_npu / rmse_gpu}")
        assert relative_max_npu / relative_max_gpu <= 2, "A_local 最大相对误差比 check failed"
        assert relative_mean_npu / relative_mean_gpu <= 1.2, "A_local 平均相对误差比 check failed"
        assert rmse_npu / rmse_gpu <= 1.2, "A_local 均方根误差比 check failed"
    
    print("All outputs match within tolerance!")
    
    # Warmup run
    q_new, k_new, o, A_local = chunk_transform_qk_fwd(
        q=q,
        k=k,
        v=v,
        beta=beta,
        A=A,
        scale=scale,
        chunk_size=chunk_size,
        output_attentions=output_attentions,
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
            q_new, k_new, o, A_local = chunk_transform_qk_fwd(
                q=q,
                k=k,
                v=v,
                beta=beta,
                A=A,
                scale=scale,
                chunk_size=chunk_size,
                output_attentions=output_attentions,
            )
            torch.npu.synchronize()
            prof.step()
        prof.stop()
    
    print(f"Test passed! q_new shape: {q_new.shape}, k_new shape: {k_new.shape}, o shape: {o.shape}")
    print(f"Profiling results saved to ./result_dir_original")


if __name__ == "__main__":
    test_chunk_transform_qk_fwd_kernel()
