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


# Copy kernel definition from original file
@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_delta_rule_fwd_kernel(
    q,
    k,
    v,
    u,
    beta,
    o,
    h0,
    ht,
    cu_seqlens,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_BETA_HEADWISE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_k, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    p_q = q + (bos * H + i_h) * K + i_k * BK + tl.arange(0, BK)
    p_k = k + (bos * H + i_h) * K + i_k * BK + tl.arange(0, BK)
    p_v = v + (bos * H + i_h) * V + i_v * BV + tl.arange(0, BV)
    p_u = u + (bos * H + i_h) * V + i_v * BV + tl.arange(0, BV)
    if IS_BETA_HEADWISE:
        p_beta = beta + (bos * H + i_h) * V + i_v * BV + tl.arange(0, BV)
    else:
        p_beta = beta + bos * H + i_h
    p_o = o + ((i_k * all + bos) * H + i_h) * V + i_v * BV + tl.arange(0, BV)

    mask_k = (i_k * BK + tl.arange(0, BK)) < K
    mask_v = (i_v * BV + tl.arange(0, BV)) < V
    mask_h = mask_k[None, :] & mask_v[:, None]

    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh * K * V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for _ in range(0, T):
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32) * scale
        b_v_minus = tl.sum(b_h * b_k[None, :], axis=1)
        b_v -= b_v_minus
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)
        else:
            b_beta = tl.load(p_beta).to(tl.float32)
        tl.store(p_u, b_v.to(p_v.dtype.element_ty), mask=mask_v)
        b_v *= b_beta
        b_h += b_k[None, :] * b_v[:, None]
        b_o = b_h * b_q[None, :]
        b_o = tl.sum(b_o, axis=1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        p_q += H*K
        p_k += H*K
        p_o += H*V
        p_v += H*V
        p_u += H*V
        p_beta += H * (V if IS_BETA_HEADWISE else 1)

    if STORE_FINAL_STATE:
        p_ht = ht + i_nh * K * V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


# Copy call function from original file
def fused_recurrent_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 8)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = 1
    num_warps = 1

    o = q.new_empty(NK, *v.shape)
    if output_final_state:
        final_state = q.new_empty(N, H, K, V, dtype=torch.float32)
    else:
        final_state = None

    grid = (NV, NK, N * H)
    u = torch.empty_like(v)
    fused_recurrent_delta_rule_fwd_kernel[grid](
        q,
        k,
        v,
        u,
        beta,
        o,
        initial_state,
        final_state,
        cu_seqlens,
        scale,
        T=T,
        B=B,
        H=H,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        IS_BETA_HEADWISE=beta.ndim == v.ndim,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    o = o.squeeze(0)
    return o, u, final_state


def test_fused_recurrent_delta_rule_fwd_kernel():
    # Config with large shape for performance testing
    B, T, H, K, V = 4, 128, 8, 128, 128
    
    # Load inputs from saved pt file
    device = torch.device('npu:0')
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_vv_ops_2/fused_recurrent_delta_rule_fwd_kernel'
    inputs_dict = torch.load(os.path.join(load_dir, 'inputs.pt'))
    
    # Convert inputs to npu
    q = inputs_dict['q'].to(device)
    k = inputs_dict['k'].to(device)
    v = inputs_dict['v'].to(device)
    beta = inputs_dict['beta'].to(device)
    scale = inputs_dict['scale']
    initial_state = inputs_dict['initial_state'].to(device)
    output_final_state = inputs_dict['output_final_state']
    
    print(f"Inputs loaded from {load_dir}/inputs.pt")
    
    # Execute kernel once for correctness check
    o, u, final_state = fused_recurrent_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
    )
    torch.npu.synchronize()
    
    # Load expected outputs and compare
    expected_outputs = torch.load(os.path.join(load_dir, 'outputs.pt'))
    expected_outputs_fp64 = torch.load(os.path.join(load_dir, 'outputs_fp64.pt'))
    
    # Convert expected outputs to npu for comparison
    expected_o = expected_outputs['o'].to(device)
    expected_u = expected_outputs['u'].to(device)
    expected_final_state = expected_outputs['final_state'].to(device) if expected_outputs['final_state'] is not None else None
    
    expected_o_fp64 = expected_outputs_fp64['o_fp64'].to(device)
    expected_u_fp64 = expected_outputs_fp64['u_fp64'].to(device)
    expected_final_state_fp64 = expected_outputs_fp64['final_state_fp64'].to(device) if expected_outputs_fp64['final_state_fp64'] is not None else None
    
    # Compare outputs with new precision logic
    print("\nComparing o outputs...")
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(o, expected_o_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(expected_o, expected_o_fp64)
    print(f"最大相对误差比 :{relative_max_npu / relative_max_gpu}, 平均相对误差比 :{relative_mean_npu / relative_mean_gpu}, 均方根误差比 :{rmse_npu / rmse_gpu}")
    print(f"relative_max_npu: {relative_max_npu}; relative_mean_npu: {relative_mean_npu}; rmse_npu: {rmse_npu}")
    print(f"relative_max_gpu: {relative_max_gpu}; relative_mean_gpu: {relative_mean_gpu}; rmse_gpu: {rmse_gpu}")
    assert relative_max_npu / relative_max_gpu <= 2, "o 最大相对误差比 check failed"
    assert relative_mean_npu / relative_mean_gpu <= 1.2, "o 平均相对误差比 check failed"
    assert rmse_npu / rmse_gpu <= 1.2, "o 均方根误差比 check failed"
    
    print("\nComparing u outputs...")
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(u, expected_u_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(expected_u, expected_u_fp64)
    print(f"最大相对误差比 :{relative_max_npu / relative_max_gpu}, 平均相对误差比 :{relative_mean_npu / relative_mean_gpu}, 均方根误差比 :{rmse_npu / rmse_gpu}")
    assert relative_max_npu / relative_max_gpu <= 2, "u 最大相对误差比 check failed"
    assert relative_mean_npu / relative_mean_gpu <= 1.2, "u 平均相对误差比 check failed"
    assert rmse_npu / rmse_gpu <= 1.2, "u 均方根误差比 check failed"
    
    if expected_final_state is not None and expected_final_state_fp64 is not None:
        print("\nComparing final_state outputs...")
        relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(final_state, expected_final_state_fp64)
        relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(expected_final_state, expected_final_state_fp64)
        print(f"最大相对误差比 :{relative_max_npu / relative_max_gpu}, 平均相对误差比 :{relative_mean_npu / relative_mean_gpu}, 均方根误差比 :{rmse_npu / rmse_gpu}")
        assert relative_max_npu / relative_max_gpu <= 2, "final_state 最大相对误差比 check failed"
        assert relative_mean_npu / relative_mean_gpu <= 1.2, "final_state 平均相对误差比 check failed"
        assert rmse_npu / rmse_gpu <= 1.2, "final_state 均方根误差比 check failed"
    
    print("All outputs match within tolerance!")
    
    # Warmup run
    o, u, final_state = fused_recurrent_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
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
            o, u, final_state = fused_recurrent_delta_rule_fwd(
                q=q,
                k=k,
                v=v,
                beta=beta,
                scale=scale,
                initial_state=initial_state,
                output_final_state=output_final_state,
            )
            torch.npu.synchronize()
            prof.step()
        prof.stop()
    
    print(f"Test passed! Output shape: {o.shape}")
    print(f"Profiling results saved to ./result_dir_original")


if __name__ == "__main__":
    test_fused_recurrent_delta_rule_fwd_kernel()
