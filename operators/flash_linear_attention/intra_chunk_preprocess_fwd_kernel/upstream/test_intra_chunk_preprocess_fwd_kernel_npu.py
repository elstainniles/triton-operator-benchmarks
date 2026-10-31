# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import torch.nn.functional as F
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


# ============ 依赖函数定义 ============
def tensor_cache(fn):
    """Simple cache decorator for tensor operations"""
    cache = {}
    def wrapper(*args, **kwargs):
        key_parts = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                key_parts.append((arg.data_ptr(), tuple(arg.shape), arg.dtype))
            else:
                key_parts.append(arg)
        key = tuple(key_parts)
        
        if key not in cache:
            cache[key] = fn(*args, **kwargs)
        return cache[key]
    return wrapper


@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return torch.diff(cu_seqlens)


@tensor_cache
def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
    cu_seqlens_cpu: torch.LongTensor | None = None,
) -> torch.LongTensor:
    if cu_seqlens_cpu is not None:
        indices = torch.cat([torch.arange(n, device=cu_seqlens.device)
                            for n in triton.cdiv(prepare_lens(cu_seqlens_cpu), chunk_size).tolist()])
        return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)
    indices = torch.cat([torch.arange(n) for n in triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


# ============ Kernel 定义 ============
@triton.heuristics({
    "USE_G": lambda args: args['g_cumsum'] is not None,
    "IS_VARLEN": lambda args: args['offsets'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def intra_chunk_preprocess_fwd_kernel(
    q,
    k,
    v,
    w,
    beta,
    g_cumsum,
    o,
    A,
    L,
    M,
    w2,
    q_new,
    k_new,
    scale,
    indices,  # varlen helper
    offsets,  # varlen helper
    T,
    H: tl.constexpr,
    G: tl.constexpr,
    HQ: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_G: tl.constexpr,
):
    i_t, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hq = i_nh // HQ, i_nh % HQ
    i_h = i_hq // G

    if IS_VARLEN:
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T

    sm_scale = scale * 1.44269504
    # offset calculations
    A += (bos*H + i_h) * BT
    q += (bos*HQ + i_hq) * K
    q_new += (bos*HQ + i_hq) * K
    k += (bos*H + i_h) * K
    k_new += (bos*H + i_h) * K
    w2 += (bos*H + i_h) * K
    w += (bos*H + i_h) * K
    v += (bos*H + i_h) * V
    o += (bos*HQ + i_hq) * V
    beta += (bos*H + i_h)
    if USE_G:
        g_cumsum += (bos*HQ + i_hq)
    L += (bos*HQ + i_hq)
    M += (bos*HQ + i_hq)

    p_q = tl.make_block_ptr(q, (T, K), (HQ*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k, (K, T), (1, H*K), (0, i_t * BT), (BK, BT), (0, 1))
    p_w = tl.make_block_ptr(w, (T, K), (H*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_v = tl.make_block_ptr(v, (T, V), (H*V, 1), (i_t * BT, 0), (BT, BV), (1, 0))
    p_beta = tl.make_block_ptr(beta, (T, ), (H, ), (i_t * BT, ), (BT, ), (0, ))
    p_T = tl.make_block_ptr(A, (T, BT), (BT*H, 1), (i_t * BT, 0), (BT, BT), (1, 0))

    b_beta = tl.load(p_beta, boundary_check=(0, ))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_kt = tl.load(p_k, boundary_check=(0, 1))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_w = tl.load(p_w, boundary_check=(0, 1))
    b_T = tl.load(p_T, boundary_check=(0, 1))
    b_T = b_T * b_beta[None, :]

    o_i = tl.arange(0, BT)
    m_t = o_i[:, None] >= o_i[None, :]

    b_qw = tl.where(m_t, tl.dot(b_q, tl.trans(b_w.to(b_q.dtype))), 0).to(b_q.dtype)
    b_qwT = tl.dot(b_qw, b_T.to(b_q.dtype)).to(b_q.dtype)
    b_wbk = tl.where(o_i[:, None] > o_i[None, :], tl.dot(b_w.to(b_q.dtype), b_kt), 0).to(b_q.dtype)
    b_A = tl.where(m_t, tl.dot(b_q, b_kt) - tl.dot(b_qwT.to(b_q.dtype), b_wbk), 0)

    b_q = b_q.to(tl.float32) - tl.dot(b_qwT, b_w.to(b_q.dtype))
    p_q_new = tl.make_block_ptr(q_new, (T, K), (K*HQ, 1), (i_t * BT, 0), (BT, K), (1, 0))
    tl.store(p_q_new, b_q.to(p_q_new.dtype.element_ty), boundary_check=(0, 1))

    if i_hq % G == 0:
        b_Twb = tl.dot(b_T, b_w)
        p_w2 = tl.make_block_ptr(w2, (T, K), (K*H, 1), (i_t * BT, 0), (BT, BK), (1, 0))
        tl.store(p_w2, b_Twb.to(p_w2.dtype.element_ty), boundary_check=(0, 1))
        b_T_wbk = tl.dot(b_T.to(b_kt.dtype), b_wbk).to(b_kt.dtype)
        p_k_new = tl.make_block_ptr(k_new, (K, T), (1, K*H), (0, i_t * BT), (BK, BT), (0, 1))
        tl.store(p_k_new, (b_kt - tl.dot(tl.trans(b_w.to(b_kt.dtype)), b_T_wbk)
                           ).to(p_k_new.dtype.element_ty), boundary_check=(0, 1))

    if USE_G:
        p_g_cumsum = tl.make_block_ptr(g_cumsum, (T, ), (HQ, ), (i_t * BT, ), (BT, ), (0, ))
        b_g_cumsum = tl.load(p_g_cumsum, boundary_check=(0, ))
        b_A = b_A + (b_g_cumsum[:, None] - b_g_cumsum[None, :])
        b_A = tl.where((i_t * BT + tl.arange(0, BT) < T)[:, None], b_A, float("-inf"))  # avoid nan

    b_qkT_softmax = tl.where(o_i[:, None] >= o_i[None, :], b_A * sm_scale, float("-inf"))
    m_i = tl.max(b_qkT_softmax, 1)
    b_qkT_softmax = tl.math.exp2(b_qkT_softmax - m_i[:, None])
    l_i = tl.sum(b_qkT_softmax, 1)
    b_o = tl.dot(b_qkT_softmax.to(b_v.dtype), b_v)
    p_o = tl.make_block_ptr(o, (T, V), (V*HQ, 1), (i_t * BT, 0), (BT, BV), (1, 0))
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    p_l = tl.make_block_ptr(L, (T, ), (HQ, ), (i_t * BT, ), (BT, ), (0, ))
    p_m = tl.make_block_ptr(M, (T, ), (HQ, ), (i_t * BT, ), (BT, ), (0, ))
    tl.store(p_m, m_i.to(p_m.dtype.element_ty), boundary_check=(0,))
    tl.store(p_l, l_i.to(p_l.dtype.element_ty), boundary_check=(0,))


# ============ Call Function 定义 ============
def intra_chunk_preprocess_fwd_fn(q, k, v, w, beta, g_cumsum, A, scale, BT, cu_seqlens,
                                  chunk_indices: torch.LongTensor | None = None):
    HQ = q.shape[-2]
    B, T, H, K = k.shape
    V = v.shape[-1]
    q_new = torch.empty_like(q, dtype=torch.float32)  # for stability
    k_new = torch.empty_like(k)
    o = torch.empty(B, T, HQ, V, device=q.device, dtype=torch.float32)

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    indices = chunk_indices
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(indices)
    grid = (NT, B*HQ)
    L = torch.empty(B, T, HQ, dtype=torch.float32, device=q.device)
    M = torch.empty(B, T, HQ, dtype=torch.float32, device=q.device)
    w2 = torch.empty_like(w)
    G = HQ//H

    intra_chunk_preprocess_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        w=w,
        beta=beta,
        g_cumsum=g_cumsum,
        o=o,
        A=A,
        L=L,
        M=M,
        w2=w2,
        q_new=q_new,
        k_new=k_new,
        scale=scale,
        offsets=cu_seqlens,
        indices=indices,
        T=T,
        H=H,
        G=G,
        HQ=HQ,
        K=K,
        V=V,
        BK=triton.next_power_of_2(K),
        BV=triton.next_power_of_2(V),
        BT=BT,
        num_warps=4 if BT == 64 else 2,
    )
    return q_new, k_new, w2, o, L, M


# ============ 测试用例 ============
def test_intra_chunk_preprocess_fwd_kernel():
    # 配置参数 - 使用较大的shape
    B = 2
    T = 1024
    H = 8
    HQ = 16  # HQ = H * G, G = 2
    K = 64
    V = 64
    BT = 64
    
    device = torch.device('npu:0')
    dtype = torch.float16
    
    # Load input tensors from pt file
    input_dir = "/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/intra_chunk_preprocess_fwd_kernel"
    print(f"Loading inputs from: {input_dir}/inputs.pt")
    
    inputs_dict = torch.load(os.path.join(input_dir, "inputs.pt"))
    
    q = inputs_dict['q'].to(device)
    k = inputs_dict['k'].to(device)
    v = inputs_dict['v'].to(device)
    w = inputs_dict['w'].to(device)
    beta = inputs_dict['beta'].to(device)
    g_cumsum = inputs_dict['g_cumsum'].to(device)
    A = inputs_dict['A'].to(device)
    scale = inputs_dict['scale']
    BT = inputs_dict['BT']
    
    print(f"测试配置:")
    print(f"  B={B}, T={T}, H={H}, HQ={HQ}, K={K}, V={V}, BT={BT}")
    print(f"  q shape: {q.shape}")
    print(f"  k shape: {k.shape}")
    print(f"  v shape: {v.shape}")
    print(f"  w shape: {w.shape}")
    print(f"  beta shape: {beta.shape}")
    print(f"  g_cumsum shape: {g_cumsum.shape}")
    print(f"  A shape: {A.shape}")
    print(f"  scale: {scale}")
    
    # Execute the kernel once
    print("\nExecuting kernel...")
    q_new_npu, k_new_npu, w2_npu, o_npu, L_npu, M_npu = intra_chunk_preprocess_fwd_fn(
        q=q, k=k, v=v, w=w, beta=beta, g_cumsum=g_cumsum,
        A=A, scale=scale, BT=BT, cu_seqlens=None
    )
    torch.npu.synchronize()
    
    # Load expected outputs and compare
    print("\nLoading expected outputs and comparing...")
    outputs_dict = torch.load(os.path.join(input_dir, "outputs.pt"))
    
    q_new_gpu_fp64 = outputs_dict['q_new_fp64'].to(device)
    k_new_gpu_fp64 = outputs_dict['k_new_fp64'].to(device)
    w2_gpu_fp64 = outputs_dict['w2_fp64'].to(device)
    o_gpu_fp64 = outputs_dict['o_fp64'].to(device)
    L_gpu_fp64 = outputs_dict['L_fp64'].to(device)
    M_gpu_fp64 = outputs_dict['M_fp64'].to(device)
    
    q_new_gpu = outputs_dict['q_new'].to(device)
    k_new_gpu = outputs_dict['k_new'].to(device)
    w2_gpu = outputs_dict['w2'].to(device)
    o_gpu = outputs_dict['o'].to(device)
    L_gpu = outputs_dict['L'].to(device)
    M_gpu = outputs_dict['M'].to(device)
    
    # Compare outputs with precision check
    print("\nq_new comparison:")
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(q_new_npu, q_new_gpu_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(q_new_gpu, q_new_gpu_fp64)
    print(f"最大相对误差比 :{relative_max_npu / relative_max_gpu}, 平均相对误差比 :{relative_mean_npu / relative_mean_gpu}, 均方根误差比 :{rmse_npu / rmse_gpu}")
    assert relative_max_npu / relative_max_gpu <= 2, "q_new 最大相对误差比 check failed"
    assert relative_mean_npu / relative_mean_gpu <= 1.2, "q_new 平均相对误差比 check failed"
    assert rmse_npu / rmse_gpu <= 1.2, "q_new 均方根误差比 check failed"
    
    print("\nk_new comparison:")
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(k_new_npu, k_new_gpu_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(k_new_gpu, k_new_gpu_fp64)
    print(f"最大相对误差比 :{relative_max_npu / relative_max_gpu}, 平均相对误差比 :{relative_mean_npu / relative_mean_gpu}, 均方根误差比 :{rmse_npu / rmse_gpu}")
    assert relative_max_npu / relative_max_gpu <= 2, "k_new 最大相对误差比 check failed"
    assert relative_mean_npu / relative_mean_gpu <= 1.2, "k_new 平均相对误差比 check failed"
    assert rmse_npu / rmse_gpu <= 1.2, "k_new 均方根误差比 check failed"
    
    print("\nw2 comparison:")
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(w2_npu, w2_gpu_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(w2_gpu, w2_gpu_fp64)
    print(f"最大相对误差比 :{relative_max_npu / relative_max_gpu}, 平均相对误差比 :{relative_mean_npu / relative_mean_gpu}, 均方根误差比 :{rmse_npu / rmse_gpu}")
    assert relative_max_npu / relative_max_gpu <= 2, "w2 最大相对误差比 check failed"
    assert relative_mean_npu / relative_mean_gpu <= 1.2, "w2 平均相对误差比 check failed"
    assert rmse_npu / rmse_gpu <= 1.2, "w2 均方根误差比 check failed"
    
    print("\no comparison:")
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(o_npu, o_gpu_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(o_gpu, o_gpu_fp64)
    print(f"最大相对误差比 :{relative_max_npu / relative_max_gpu}, 平均相对误差比 :{relative_mean_npu / relative_mean_gpu}, 均方根误差比 :{rmse_npu / rmse_gpu}")
    assert relative_max_npu / relative_max_gpu <= 2, "o 最大相对误差比 check failed"
    assert relative_mean_npu / relative_mean_gpu <= 1.2, "o 平均相对误差比 check failed"
    assert rmse_npu / rmse_gpu <= 1.2, "o 均方根误差比 check failed"
    
    print("\nL comparison:")
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(L_npu, L_gpu_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(L_gpu, L_gpu_fp64)
    print(f"最大相对误差比 :{relative_max_npu / relative_max_gpu}, 平均相对误差比 :{relative_mean_npu / relative_mean_gpu}, 均方根误差比 :{rmse_npu / rmse_gpu}")
    assert relative_max_npu / relative_max_gpu <= 2, "L 最大相对误差比 check failed"
    assert relative_mean_npu / relative_mean_gpu <= 1.2, "L 平均相对误差比 check failed"
    assert rmse_npu / rmse_gpu <= 1.2, "L 均方根误差比 check failed"
    
    print("\nM comparison:")
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(M_npu, M_gpu_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(M_gpu, M_gpu_fp64)
    print(f"最大相对误差比 :{relative_max_npu / relative_max_gpu}, 平均相对误差比 :{relative_mean_npu / relative_mean_gpu}, 均方根误差比 :{rmse_npu / rmse_gpu}")
    assert relative_max_npu / relative_max_gpu <= 2, "M 最大相对误差比 check failed"
    assert relative_mean_npu / relative_mean_gpu <= 1.2, "M 平均相对误差比 check failed"
    assert rmse_npu / rmse_gpu <= 1.2, "M 均方根误差比 check failed"
    
    print("✓ All outputs comparison passed!")
    
    # Profiling测试
    print("\n开始Profiling测试...")
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
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result_dir_original")
    ) as prof:
        prof.start()
        for i in range(35):
            _ = intra_chunk_preprocess_fwd_fn(
                q=q, k=k, v=v, w=w, beta=beta, g_cumsum=g_cumsum,
                A=A, scale=scale, BT=BT, cu_seqlens=None
            )
            torch.npu.synchronize()
            prof.step()
        prof.stop()
    
    print("\nProfiling完成，结果保存在 ./result_dir_original")
    print("测试成功完成！")


if __name__ == "__main__":
    test_intra_chunk_preprocess_fwd_kernel()
