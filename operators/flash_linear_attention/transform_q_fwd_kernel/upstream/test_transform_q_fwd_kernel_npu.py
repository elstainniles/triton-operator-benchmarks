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


@tensor_cache
def get_max_num_splits(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
    cu_seqlens_cpu: torch.LongTensor | None = None
) -> int:
    if cu_seqlens_cpu is not None:
        return triton.cdiv(int(max(prepare_lens(cu_seqlens_cpu))), chunk_size)
    return triton.cdiv(int(max(prepare_lens(cu_seqlens))), chunk_size)


# ============ Kernel 定义 ============
@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def transform_q_fwd_kernel(
    q,
    q_new,
    w1,
    w2,
    cu_seqlens,
    indices,
    T,
    S: tl.constexpr,
    G: tl.constexpr,
    HQ: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
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
        # boh = i_n * tl.cdiv(T, BS)
    p_q = tl.make_block_ptr(q + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    b_q = tl.zeros([BT, BK], dtype=tl.float32)
    b_q += tl.load(p_q, boundary_check=(0, 1))

    if BS == BT:
        if (i_t * BT) % S == 0:
            p_q_new = tl.make_block_ptr(q_new + ((bos.to(tl.int64) * NUM_BLOCKS + (i_t * BT // S)) * HQ + i_hq) * K,
                                        (T, K), (HQ*K*NUM_BLOCKS, 1), (i_t * BT, 0), (BT, BK), (1, 0))
            tl.store(p_q_new, b_q.to(q_new.dtype.element_ty), boundary_check=(0, 1))

    for offset in range((i_t + 1) * BT - 2 * BS, S-BS, -BS):
        p_w1 = tl.make_block_ptr(w1 + (bos * H + i_h) * K, (K, T), (1, K*H), (0, offset), (BK, BS), (0, 1))
        p_w2 = tl.make_block_ptr(w2 + (bos * H + i_h) * K, (T, K), (K*H, 1), (offset, 0), (BS, BK), (1, 0))
        b_w1 = tl.load(p_w1, boundary_check=(0, 1))
        b_w2 = tl.load(p_w2, boundary_check=(0, 1))
        m_s = i_t * BT + tl.arange(0, BT) >= (offset + BS)
        b_s2 = tl.dot(b_q.to(b_w1.dtype), b_w1)
        b_s2 = tl.where(m_s[:, None], b_s2, 0)
        b_q -= tl.dot(b_s2.to(b_w2.dtype), b_w2)

        if offset % S == 0:
            p_q_new = tl.make_block_ptr(q_new + ((bos.to(tl.int64) * NUM_BLOCKS + (offset // S)) * HQ + i_hq) * K,
                                        (T, K), (HQ*K*NUM_BLOCKS, 1), (i_t * BT, 0), (BT, BK), (1, 0))
            tl.store(p_q_new, b_q.to(q_new.dtype.element_ty), boundary_check=(0, 1))


# ============ Call Function 定义 ============
def transform_q_fwd_fn(
    q,
    w1,
    w2,
    cu_seqlens,
    BT,
    BS,
    S,
    chunk_indices: torch.LongTensor | None = None,
):
    B, T, HQ, K = q.shape
    H = w1.shape[-2]
    G = HQ // H
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    indices = chunk_indices
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(indices)

    num_blocks = triton.cdiv(T, S) if cu_seqlens is None else get_max_num_splits(cu_seqlens, S)
    q_new = torch.zeros(B, T, num_blocks, HQ, K, dtype=q.dtype, device=q.device)
    transform_q_fwd_kernel[(NT, B * HQ)](
        q=q,
        q_new=q_new,
        w1=w1,
        w2=w2,
        cu_seqlens=cu_seqlens,
        indices=indices,
        T=T,
        K=K,
        BK=triton.next_power_of_2(K),
        G=G,
        HQ=HQ,
        H=H,
        BS=BS,
        BT=BT,
        S=S,
        NUM_BLOCKS=num_blocks,
        num_warps=8 if (BT == 128 and K == 128) else 4,
    )
    return q_new


# ============ 测试用例 ============
def test_transform_q_fwd_kernel():
    # 配置参数 - 使用较大的shape
    B = 2
    T = 256
    H = 8
    HQ = 16  # HQ = H * G, G = 2
    K = 32
    S = 64  # larger chunk size
    BT = 32
    BS = 32  # BS == BT for this kernel
    
    device = torch.device('npu:0')
    dtype = torch.float16
    
    # Load input tensors from pt file
    import os
    load_dir = "/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/transform_q_fwd_kernel"
    inputs_dict = torch.load(os.path.join(load_dir, "inputs.pt"))
    
    # Convert loaded tensors to NPU
    q = inputs_dict['q'].to(device)
    w1 = inputs_dict['w1'].to(device)
    w2 = inputs_dict['w2'].to(device)
    cu_seqlens = inputs_dict['cu_seqlens']
    BT = inputs_dict['BT']
    BS = inputs_dict['BS']
    S = inputs_dict['S']
    
    num_blocks = triton.cdiv(T, S)
    
    print(f"测试配置:")
    print(f"  B={B}, T={T}, H={H}, HQ={HQ}, K={K}")
    print(f"  S={S}, BT={BT}, BS={BS}, num_blocks={num_blocks}")
    print(f"  q shape: {q.shape}")
    print(f"  w1 shape: {w1.shape}")
    print(f"  w2 shape: {w2.shape}")
    
    # Execute kernel once
    print("\nExecuting kernel...")
    q_new_npu = transform_q_fwd_fn(
        q=q, w1=w1, w2=w2, cu_seqlens=cu_seqlens, BT=BT, BS=BS, S=S
    )
    
    # Load expected output tensors from pt file
    outputs_dict = torch.load(os.path.join(load_dir, "outputs.pt"))
    q_new_gpu_fp64 = outputs_dict['q_new_fp64'].to(device)
    q_new_gpu = outputs_dict['q_new'].to(device)
    
    # Compare outputs with expected values
    print("\nComparing outputs with expected values...")
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(q_new_npu, q_new_gpu_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(q_new_gpu, q_new_gpu_fp64)
    print(f"最大相对误差比 :{relative_max_npu / relative_max_gpu}, 平均相对误差比 :{relative_mean_npu / relative_mean_gpu}, 均方根误差比 :{rmse_npu / rmse_gpu}")
    assert relative_max_npu / relative_max_gpu <= 2, "最大相对误差比 check failed"
    assert relative_mean_npu / relative_mean_gpu <= 1.2, "平均相对误差比 check failed"
    assert rmse_npu / rmse_gpu <= 1.2, "均方根误差比 check failed"
    # torch.testing.assert_close(q_new, q_new_expected, atol=0.008, rtol=0.008)
    print("Output comparison passed!")
    
    # 预热运行
    print("\n预热运行...")
    _ = transform_q_fwd_fn(
        q=q, w1=w1, w2=w2, cu_seqlens=cu_seqlens, BT=BT, BS=BS, S=S
    )
    torch.npu.synchronize()
    
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
            _ = transform_q_fwd_fn(
                q=q, w1=w1, w2=w2, cu_seqlens=cu_seqlens, BT=BT, BS=BS, S=S
            )
            torch.npu.synchronize()
            prof.step()
        prof.stop()
    
    print("\nProfiling完成，结果保存在 ./result_dir_original")
    print("测试成功完成！")


if __name__ == "__main__":
    test_transform_q_fwd_kernel()
