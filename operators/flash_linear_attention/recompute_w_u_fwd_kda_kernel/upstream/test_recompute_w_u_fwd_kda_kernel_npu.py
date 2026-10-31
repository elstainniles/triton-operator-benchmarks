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
    'STORE_QG': lambda args: args['qg'] is not None,
    'STORE_KG': lambda args: args['kg'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'HV', 'K', 'V', 'BT', 'BK', 'BV', 'IS_VARLEN'],
)
@triton.jit(do_not_specialize=['T'])
def recompute_w_u_fwd_kda_kernel(
    q,
    k,
    qg,
    kg,
    v,
    beta,
    w,
    u,
    A,
    gk,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    STORE_QG: tl.constexpr,
    STORE_KG: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    k += (bos * H + i_h) * K
    v += (bos * HV + i_hv) * V
    u += (bos * HV + i_hv) * V
    w += (bos * HV + i_hv) * K
    gk += (bos * HV + i_hv) * K
    beta += bos * HV + i_hv
    A += (bos * HV + i_hv) * BT
    if STORE_QG:
        q += (bos * H + i_h) * K
        qg += (bos * HV + i_hv) * K
    if STORE_KG:
        kg += (bos * HV + i_hv) * K

    p_b = tl.make_block_ptr(beta, (T,), (HV,), (i_t * BT,), (BT,), (0,))
    b_b = tl.load(p_b, boundary_check=(0,))

    p_A = tl.make_block_ptr(A, (T, BT), (HV*BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_A = tl.load(p_A, boundary_check=(0, 1))

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v, (T, V), (HV*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_u = tl.make_block_ptr(u, (T, V), (HV*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))

    for i_k in range(tl.cdiv(K, BK)):
        p_w = tl.make_block_ptr(w, (T, K), (HV*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = b_k * b_b[:, None]

        p_gk = tl.make_block_ptr(gk, (T, K), (HV*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_gk = tl.load(p_gk, boundary_check=(0, 1)).to(tl.float32)
        b_kb *= exp2(b_gk)
        if STORE_QG:
            p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_qg = tl.make_block_ptr(qg, (T, K), (HV*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_qg = b_q * exp2(b_gk)
            tl.store(p_qg, b_qg.to(p_qg.dtype.element_ty), boundary_check=(0, 1))
        if STORE_KG:
            last_idx = min(i_t * BT + BT, T) - 1
            o_k = i_k * BK + tl.arange(0, BK)
            m_k = o_k < K
            b_gn = tl.load(gk + last_idx * HV*K + o_k, mask=m_k, other=0.).to(tl.float32)
            b_kg = b_k * tl.where((i_t * BT + tl.arange(0, BT) < T)[:, None], exp2(b_gn[None, :] - b_gk), 0)
            p_kg = tl.make_block_ptr(kg, (T, K), (HV*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            tl.store(p_kg, b_kg.to(p_kg.dtype.element_ty), boundary_check=(0, 1))

        b_w = tl.dot(b_A, b_kb.to(b_k.dtype))
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))


# Copy call function from original file
def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    gk: torch.Tensor,
    q: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    BT = A.shape[-1]
    BK = 64
    BV = 64

    NT = triton.cdiv(T, BT)

    w = torch.empty(B, T, HV, K, device=k.device, dtype=k.dtype)
    u = torch.empty_like(v)
    qg = torch.empty(B, T, HV, K, device=k.device, dtype=k.dtype) if q is not None else None
    kg = torch.empty(B, T, HV, K, device=k.device, dtype=k.dtype)
    recompute_w_u_fwd_kda_kernel[(NT, B*HV)](
        q=q,
        k=k,
        qg=qg,
        kg=kg,
        v=v,
        beta=beta,
        w=w,
        u=u,
        A=A,
        gk=gk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
    )
    return w, u, qg, kg


def test_recompute_w_u_fwd_kda_kernel():
    # Config: (B, T, H, HV, D) = (2, 256, 2, 4, 64)
    # k shape: [B, T, H, K] = [2, 256, 2, 64]
    # v shape: [B, T, HV, V] = [2, 256, 4, 64]
    # beta shape: [B, T, HV] = [2, 256, 4]
    # A shape: [B, T, HV, BT] = [2, 256, 4, 64]
    # gk shape: [B, T, HV, K] = [2, 256, 4, 64]
    
    # Load inputs from saved pt file
    device = torch.device('npu:0')
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/recompute_w_u_fwd_kda_kernel'
    inputs_dict = torch.load(os.path.join(load_dir, 'inputs.pt'))
    
    # Convert inputs to npu
    k = inputs_dict['k'].to(device)
    v = inputs_dict['v'].to(device)
    beta = inputs_dict['beta'].to(device)
    A = inputs_dict['A'].to(device)
    gk = inputs_dict['gk'].to(device)
    q = inputs_dict['q']
    BT = inputs_dict['BT']
    
    print(f"Inputs loaded from {load_dir}/inputs.pt")
    print(f"k shape: {k.shape}, v shape: {v.shape}, beta shape: {beta.shape}, A shape: {A.shape}, gk shape: {gk.shape}")
    
    # Execute kernel once for correctness check
    w, u, qg, kg = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        gk=gk,
        q=q,
        cu_seqlens=None,
    )
    torch.npu.synchronize()
    
    # Load expected outputs and compare
    expected_outputs = torch.load(os.path.join(load_dir, 'outputs.pt'))
    expected_outputs_fp64 = torch.load(os.path.join(load_dir, 'outputs_fp64.pt'))
    
    # Convert expected outputs to npu for comparison
    expected_w = expected_outputs['w'].to(device)
    expected_u = expected_outputs['u'].to(device)
    expected_qg = expected_outputs['qg'].to(device) if expected_outputs['qg'] is not None else None
    expected_kg = expected_outputs['kg'].to(device)
    
    expected_w_fp64 = expected_outputs_fp64['w_fp64'].to(device)
    expected_u_fp64 = expected_outputs_fp64['u_fp64'].to(device)
    expected_qg_fp64 = expected_outputs_fp64['qg_fp64'].to(device) if expected_outputs_fp64['qg_fp64'] is not None else None
    expected_kg_fp64 = expected_outputs_fp64['kg_fp64'].to(device)
    
    # Compare outputs with new precision logic
    print("\nComparing w outputs...")
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(w, expected_w_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(expected_w, expected_w_fp64)
    print(f"最大相对误差比 :{relative_max_npu / relative_max_gpu}, 平均相对误差比 :{relative_mean_npu / relative_mean_gpu}, 均方根误差比 :{rmse_npu / rmse_gpu}")
    print(f"relative_max_npu: {relative_max_npu}; relative_mean_npu: {relative_mean_npu}; rmse_npu: {rmse_npu}")
    print(f"relative_max_gpu: {relative_max_gpu}; relative_mean_gpu: {relative_mean_gpu}; rmse_gpu: {rmse_gpu}")
    assert relative_max_npu / relative_max_gpu <= 2, "w 最大相对误差比 check failed"
    assert relative_mean_npu / relative_mean_gpu <= 1.2, "w 平均相对误差比 check failed"
    assert rmse_npu / rmse_gpu <= 1.2, "w 均方根误差比 check failed"
    
    print("\nComparing u outputs...")
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(u, expected_u_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(expected_u, expected_u_fp64)
    print(f"最大相对误差比 :{relative_max_npu / relative_max_gpu}, 平均相对误差比 :{relative_mean_npu / relative_mean_gpu}, 均方根误差比 :{rmse_npu / rmse_gpu}")
    assert relative_max_npu / relative_max_gpu <= 2, "u 最大相对误差比 check failed"
    assert relative_mean_npu / relative_mean_gpu <= 1.2, "u 平均相对误差比 check failed"
    assert rmse_npu / rmse_gpu <= 1.2, "u 均方根误差比 check failed"
    
    print("\nComparing kg outputs...")
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(kg, expected_kg_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(expected_kg, expected_kg_fp64)
    print(f"最大相对误差比 :{relative_max_npu / relative_max_gpu}, 平均相对误差比 :{relative_mean_npu / relative_mean_gpu}, 均方根误差比 :{rmse_npu / rmse_gpu}")
    assert relative_max_npu / relative_max_gpu <= 2, "kg 最大相对误差比 check failed"
    assert relative_mean_npu / relative_mean_gpu <= 1.2, "kg 平均相对误差比 check failed"
    assert rmse_npu / rmse_gpu <= 1.2, "kg 均方根误差比 check failed"
    
    print("All outputs match within tolerance!")
    
    # Warmup run
    w, u, qg, kg = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        gk=gk,
        q=q,
        cu_seqlens=None,
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
            w, u, qg, kg = recompute_w_u_fwd(
                k=k,
                v=v,
                beta=beta,
                A=A,
                gk=gk,
                q=q,
                cu_seqlens=None,
            )
            torch.npu.synchronize()
            prof.step()
        prof.stop()
    
    print(f"Test passed! w shape: {w.shape}, u shape: {u.shape}, kg shape: {kg.shape}")
    print(f"Profiling results saved to ./result_dir_original")


if __name__ == "__main__":
    test_recompute_w_u_fwd_kda_kernel()
