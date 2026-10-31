import torch
import torch_npu
import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice
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


if os.environ.get('FLA_USE_FAST_OPS', '0') == '1':
    @triton.jit
    def exp2(x): return tldevice.exp2(x.to(tl.float32))
else:
    @triton.jit
    def exp2(x): return tl.math.exp2(x.to(tl.float32))

# Copy kernel definition from original file
@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK, 'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64]
        for BV in [64, 128]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['BT'],
)
@triton.jit(do_not_specialize=['T'])
def chunk_gla_bwd_kernel_dv(
    k,
    g,
    A,
    do,
    dh,
    dv,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
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

    p_A = tl.make_block_ptr(A + (bos * H + i_h) * BT, (BT, T), (1, H*BT), (0, i_t * BT), (BT, BT), (0, 1))
    p_do = tl.make_block_ptr(do + (bos * H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_dv = tl.make_block_ptr(dv + (bos * H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_do = tl.load(p_do, boundary_check=(0, 1))

    b_A = tl.where(tl.arange(0, BT)[:, None] <= tl.arange(0, BT)[None, :], b_A, 0.)
    # (SY 09/17) important to disallow tf32 here to maintain a good precision.
    b_dv = tl.dot(b_A, b_do.to(b_A.dtype), allow_tf32=False)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K

        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_gk = tl.make_block_ptr(g + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_gn = g + (bos + min(i_t * BT + BT, T) - 1)*H*K + i_h * K + o_k
        p_dh = tl.make_block_ptr(dh + (i_tg * H + i_h) * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))

        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_gk = tl.load(p_gk, boundary_check=(0, 1))
        b_dh = tl.load(p_dh, boundary_check=(0, 1))

        b_gn = exp2(tl.load(p_gn, mask=m_k, other=0)[None, :] - b_gk)
        b_k = (b_k * b_gn).to(b_k.dtype)
        # [BT, BV]
        # (SY 09/17) it is ok to have bf16 interchunk gradient contribution here
        b_dv += tl.dot(b_k, b_dh.to(b_k.dtype))

    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


# Copy call function from original file
def chunk_gla_bwd_dv(
    k: torch.Tensor,
    g: torch.Tensor,
    A: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
):
    B, T, H, K, V = *k.shape, do.shape[-1]
    BT = chunk_size

    NT = triton.cdiv(T, BT)

    dv = torch.empty_like(do)
    def grid(meta): return (triton.cdiv(V, meta['BV']), NT, B * H)
    chunk_gla_bwd_kernel_dv[grid](
        k=k,
        g=g,
        A=A,
        do=do,
        dh=dh,
        dv=dv,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
    )
    return dv


def test_chunk_gla_bwd_kernel_dv():
    # Config from official test script: (B, T, H, D, gate_logit_normalizer, dtype) = (4, 2048, 4, 64, 1, torch.float16)
    # k shape: [B, T, H, K] = [4, 2048, 4, 64]
    # g shape: [B, T, H, K] = [4, 2048, 4, 64] (gate cumulative sum)
    # A shape: [B, T, H, BT] = [4, 2048, 4, 64] (intra-chunk attention)
    # do shape: [B, T, H, V] = [4, 2048, 4, 64] (gradient of output)
    # dh shape: [NT, B, H, K, V] = [32, 4, 4, 64, 64] (gradient of hidden state)
    # dv shape: [B, T, H, V] = [4, 2048, 4, 64] (output)
    
    # Load inputs from saved pt file
    device = torch.device('npu:0')
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_gla_bwd_kernel_dv'
    inputs_dict = torch.load(os.path.join(load_dir, 'inputs.pt'))
    
    # Convert inputs to npu
    k = inputs_dict['k'].to(device)
    g = inputs_dict['g'].to(device)
    A = inputs_dict['A'].to(device)
    do = inputs_dict['do'].to(device)
    dh = inputs_dict['dh'].to(device)
    BT = inputs_dict['BT']
    
    print(f"Inputs loaded from {load_dir}/inputs.pt")
    print(f"k shape: {k.shape}, g shape: {g.shape}, A shape: {A.shape}, do shape: {do.shape}, dh shape: {dh.shape}")
    
    # Execute kernel once for correctness check
    dv = chunk_gla_bwd_dv(
        k=k,
        g=g,
        A=A,
        do=do,
        dh=dh,
        cu_seqlens=None,
        chunk_size=BT,
    )
    torch.npu.synchronize()
    
    # Load expected outputs and compare
    expected_outputs = torch.load(os.path.join(load_dir, 'outputs.pt'))
    expected_outputs_fp64 = torch.load(os.path.join(load_dir, 'outputs_fp64.pt'))
    
    # Convert expected outputs to npu for comparison
    expected_dv = expected_outputs['dv'].to(device)
    expected_dv_fp64 = expected_outputs_fp64['dv_fp64'].to(device)
    
    # Compare outputs with new precision logic
    print("\nComparing dv outputs...")
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(dv, expected_dv_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(expected_dv, expected_dv_fp64)
    print(f"最大相对误差比 :{relative_max_npu / relative_max_gpu}, 平均相对误差比 :{relative_mean_npu / relative_mean_gpu}, 均方根误差比 :{rmse_npu / rmse_gpu}")
    print(f"relative_max_npu: {relative_max_npu}; relative_mean_npu: {relative_mean_npu}; rmse_npu: {rmse_npu}")
    print(f"relative_max_gpu: {relative_max_gpu}; relative_mean_gpu: {relative_mean_gpu}; rmse_gpu: {rmse_gpu}")
    assert relative_max_npu / relative_max_gpu <= 2, "dv 最大相对误差比 check failed"
    assert relative_mean_npu / relative_mean_gpu <= 1.2, "dv 平均相对误差比 check failed"
    assert rmse_npu / rmse_gpu <= 1.2, "dv 均方根误差比 check failed"
    
    print("All outputs match within tolerance!")
    
    # Warmup run
    dv = chunk_gla_bwd_dv(
        k=k,
        g=g,
        A=A,
        do=do,
        dh=dh,
        cu_seqlens=None,
        chunk_size=BT,
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
            dv = chunk_gla_bwd_dv(
                k=k,
                g=g,
                A=A,
                do=do,
                dh=dh,
                cu_seqlens=None,
                chunk_size=BT,
            )
            torch.npu.synchronize()
            prof.step()
        prof.stop()
    
    print(f"Test passed! dv shape: {dv.shape}")
    print(f"Profiling results saved to ./result_dir_original")


if __name__ == "__main__":
    test_chunk_gla_bwd_kernel_dv()
