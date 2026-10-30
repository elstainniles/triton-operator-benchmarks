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
import torch.nn.functional as F
import functools


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
@functools.cache
def check_shared_mem(arch: str = "none", tensor_idx: int = 0) -> bool:
    """检查共享内存是否满足要求"""
    try:
        import triton.runtime.driver.active as driver
        device_shared_mem_list = [
            driver.utils.get_device_properties(i)['max_shared_mem']
            for i in range(torch.npu.device_count())
        ]
        max_shared_memory = device_shared_mem_list[tensor_idx]
        
        class Backend:
            ADA = 101376       # RTX 4090
            AMPERE = 166912    # A100
            HOPPER = 232448    # H100
            DEFAULT = 102400   # Default
            
            @classmethod
            def get_shared_memory(cls, arch: str) -> int:
                try:
                    return cls[arch.upper()].value
                except KeyError:
                    return cls.DEFAULT.value
        
        return max_shared_memory >= Backend.get_shared_memory(arch)
    except Exception:
        return False


@functools.cache
def tensor_cache(fn):
    """简单的tensor缓存装饰器"""
    last_args = None
    last_kwargs = None
    last_result = None
    
    def wrapper(*args, **kwargs):
        nonlocal last_args, last_kwargs, last_result
        if last_args is not None and last_kwargs is not None:
            if len(args) == len(last_args) and len(kwargs) == len(last_kwargs):
                if all(a is b for a, b in zip(args, last_args)) and \
                        all(k in last_kwargs and v is last_kwargs[k] for k, v in kwargs.items()):
                    return last_result
        result = fn(*args, **kwargs)
        last_args, last_kwargs, last_result = args, kwargs, result
        return result
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
def prepare_chunk_offsets(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
) -> torch.LongTensor:
    return F.pad(triton.cdiv(prepare_lens(cu_seqlens), chunk_size), (1, 0), value=0).cumsum(-1)


# ============ Kernel定义 ============
@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit
def chunk_cumprod_householder_fwd_kernel(
    k,
    k_new,
    w1,
    w2,
    hc_suffix,
    hc_whole,
    cu_seqlens,
    split_indices,
    chunk_offsets,
    split_offsets,
    BT: tl.constexpr,  # small chunk size
    K: tl.constexpr,
    H: tl.constexpr,
    BK: tl.constexpr,
    T: tl.constexpr,
    S: tl.constexpr,  # split size, aka large chunk size
    IS_VARLEN: tl.constexpr,
):
    i_ss, i_h = tl.program_id(0), tl.program_id(1)

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

    NT_small = tl.cdiv(min(S, T-i_s*S), BT)
    stride_h = H*K*K

    # offset calculations
    hc_whole += ((boh_large + i_s) * H + i_h) * K * K
    hc_suffix += ((boh + tl.cdiv(i_s * S, BT)) * H + i_h) * K * K

    k += (bos * H + i_h) * K
    k_new += (bos * H + i_h) * K
    w1 += (bos * H + i_h) * K
    w2 += (bos * H + i_h) * K

    b_h = tl.zeros([BK, BK], dtype=tl.float32)
    for i_t_small in range(NT_small-1, -1, -1):
        p_hc_suffix = tl.make_block_ptr(hc_suffix + i_t_small * stride_h, (K, K), (K, 1), (0, 0), (BK, BK), (1, 0))
        tl.store(p_hc_suffix, b_h.to(hc_suffix.dtype.element_ty), boundary_check=(0, 1))
        p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_s * S + i_t_small * BT, 0), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_k = (b_k - tl.dot(b_k, tl.trans(b_h.to(b_k.dtype))))
        p_w1 = tl.make_block_ptr(w1, (K, T), (1, H*K), (0, i_s * S + i_t_small * BT), (BK, BT), (0, 1))
        p_w2 = tl.make_block_ptr(w2, (T, K), (H*K, 1), (i_s * S + i_t_small * BT, 0), (BT, BK), (1, 0))
        b_w1 = tl.load(p_w1, boundary_check=(0, 1))
        b_w2 = tl.load(p_w2, boundary_check=(0, 1))
        b_v_new = (b_w1 - tl.dot(b_h.to(b_w1.dtype), b_w1)).to(b_w2.dtype)
        b_h += tl.dot(b_v_new, b_w2)
        p_k_new = tl.make_block_ptr(k_new, (T, K), (H*K, 1), (i_s * S + i_t_small * BT, 0), (BT, BK), (1, 0))
        tl.store(p_k_new, b_k.to(k_new.dtype.element_ty), boundary_check=(0, 1))

    p_hc_whole = tl.make_block_ptr(hc_whole, (K, K), (K, 1), (0, 0), (BK, BK), (1, 0))
    tl.store(p_hc_whole, b_h.to(hc_whole.dtype.element_ty), boundary_check=(0, 1))


# ============ 调用函数定义 ============
def chunk_cumprod_householder_fwd_fn(
    k: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    S: int,  # split size, aka large chunk size
    BT: int,  # small chunk size
    cu_seqlens: torch.Tensor = None,
    chunk_indices: torch.LongTensor | None = None,
):
    B, T, H, K = k.shape

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, S)
    split_indices = chunk_indices
    chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT) if cu_seqlens is not None else None
    split_offsets = prepare_chunk_offsets(cu_seqlens, S) if cu_seqlens is not None else None

    if cu_seqlens is None:
        N = B
        NS = N * triton.cdiv(T, S)
        NT = N * triton.cdiv(T, BT)
    else:
        N = len(cu_seqlens) - 1
        NS = split_offsets[-1]
        NT = chunk_offsets[-1]

    grid = (NS, H)
    hc_whole = torch.empty((NS, H, K, K), device=k.device, dtype=w1.dtype)
    k_new = torch.empty_like(k, dtype=k.dtype)
    hc_suffix = torch.empty((NT, H, K, K), device=k.device, dtype=w1.dtype)
    chunk_cumprod_householder_fwd_kernel[grid](
        k=k, k_new=k_new, w1=w1, w2=w2, hc_whole=hc_whole, hc_suffix=hc_suffix,
        cu_seqlens=cu_seqlens,
        split_indices=split_indices, chunk_offsets=chunk_offsets, split_offsets=split_offsets,
        BT=BT, K=K, H=H, BK=K,
        T=T, S=S,
        # SY (2025/07/08): I don't know why when K == 128 if I set num_warps=4 the result would be completely wrong
        num_warps=8 if K == 128 else 4,
        num_stages=3 if check_shared_mem('ampere') else 1,
    )
    return k_new, hc_suffix, hc_whole


# ============ 测试用例 ============
def test_chunk_cumprod_householder_fwd_kernel():
    """测试chunk_cumprod_householder_fwd_kernel"""
    device = torch.device('npu:0')
    
    # Load input tensors from pt files
    load_dir = "/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_cumprod_householder_fwd_kernel"
    print(f"Loading inputs from: {load_dir}")
    
    data_dict = torch.load(f"{load_dir}/chunk_cumprod_householder_fwd_kernel.pt")
    input_dict = data_dict["input_data"]
    k = input_dict["k"].to(device)
    w1 = input_dict["w1"].to(device)
    w2 = input_dict["w2"].to(device)
    S = input_dict["S"]
    BT = input_dict["BT"]
    
    print(f"测试配置: k={k.shape}, w1={w1.shape}, w2={w2.shape}, S={S}, BT={BT}")
    
    # 执行算子
    k_new, hc_suffix, hc_whole = chunk_cumprod_householder_fwd_fn(
        k=k, w1=w1, w2=w2, S=S, BT=BT
    )
    torch.npu.synchronize()
    
    # Load expected outputs and compare
    print("\nLoading expected outputs and comparing...")
    output_dict = data_dict["output_data"]
    expected_k_new = output_dict["k_new"].to(device)
    expected_hc_suffix = output_dict["hc_suffix"].to(device)
    expected_hc_whole = output_dict["hc_whole"].to(device)

    output_fp64_dict = data_dict["output_data_fp64"]
    expected_k_new_fp64 = output_fp64_dict["k_new_fp64"].to(device)
    expected_hc_suffix_fp64 = output_fp64_dict["hc_suffix_fp64"].to(device)
    expected_hc_whole_fp64 = output_fp64_dict["hc_whole_fp64"].to(device)
    
    # Compare outputs with new precision logic
    print("\nComparing k_new outputs...")
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(k_new, expected_k_new_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(expected_k_new, expected_k_new_fp64)
    print(f"最大相对误差比 :{relative_max_npu / relative_max_gpu}, 平均相对误差比 :{relative_mean_npu / relative_mean_gpu}, 均方根误差比 :{rmse_npu / rmse_gpu}")
    assert relative_max_npu / relative_max_gpu <= 2, "k_new 最大相对误差比 check failed"
    assert relative_mean_npu / relative_mean_gpu <= 1.2, "k_new 平均相对误差比 check failed"
    assert rmse_npu / rmse_gpu <= 1.2, "k_new 均方根误差比 check failed"
    
    print("\nComparing hc_suffix outputs...")
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(hc_suffix, expected_hc_suffix_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(expected_hc_suffix, expected_hc_suffix_fp64)
    print(f"最大相对误差比 :{relative_max_npu / relative_max_gpu}, 平均相对误差比 :{relative_mean_npu / relative_mean_gpu}, 均方根误差比 :{rmse_npu / rmse_gpu}")
    assert relative_max_npu / relative_max_gpu <= 2, "hc_suffix 最大相对误差比 check failed"
    assert relative_mean_npu / relative_mean_gpu <= 1.2, "hc_suffix 平均相对误差比 check failed"
    assert rmse_npu / rmse_gpu <= 1.2, "hc_suffix 均方根误差比 check failed"
    
    print("\nComparing hc_whole outputs...")
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(hc_whole, expected_hc_whole_fp64)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(expected_hc_whole, expected_hc_whole_fp64)
    print(f"最大相对误差比 :{relative_max_npu / relative_max_gpu}, 平均相对误差比 :{relative_mean_npu / relative_mean_gpu}, 均方根误差比 :{rmse_npu / rmse_gpu}")
    assert relative_max_npu / relative_max_gpu <= 2, "hc_whole 最大相对误差比 check failed"
    assert relative_mean_npu / relative_mean_gpu <= 1.2, "hc_whole 平均相对误差比 check failed"
    assert rmse_npu / rmse_gpu <= 1.2, "hc_whole 均方根误差比 check failed"
    
    print("All outputs match within tolerance!")
    
    print(f"\n输出形状: k_new={k_new.shape}, hc_suffix={hc_suffix.shape}, hc_whole={hc_whole.shape}")
    print("精度测试通过！")

    print(f"\n开始Profiling测试...")
    profiling_output_dict = "./result_dir_original"
    # 配置profiler
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
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profiling_output_dict)
    ) as prof:
        prof.start()
        for _ in range(35):
            _ = chunk_cumprod_householder_fwd_fn(k=k, w1=w1, w2=w2, S=S, BT=BT)
            torch.npu.synchronize()  # 确保 kernel 真正执行完
            prof.step()
        prof.stop()
    
    print(f"Profiling完成！结果已保存到 {profiling_output_dict}")
    
    return


# ============ Profiling测试 ============
def profile_chunk_cumprod_householder_fwd_kernel():
    """对chunk_cumprod_householder_fwd_kernel进行profiling"""
    # 配置参数 - 使用较大的shape以便测试性能
    B = 1
    T = 512
    H = 4
    K = 32
    S = 128  # split size (large chunk size)
    BT = 32  # small chunk size
    
    device = torch.device('npu:0')
    dtype = torch.float16
    
    # Load input tensors from pt files
    input_dir = "/data/s00856262/ai_agent/FLA/kernel_inputs_and_outputs/chunk_cumprod_householder_fwd_kernel"
    
    k = torch.load(f"{input_dir}/input_k.pt").to(device)
    w1 = torch.load(f"{input_dir}/input_w1.pt").to(device)
    w2 = torch.load(f"{input_dir}/input_w2.pt").to(device)
    
    print(f"\n开始Profiling测试...")
    print(f"测试配置: B={B}, T={T}, H={H}, K={K}, S={S}, BT={BT}")
    
    # 配置profiler
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
            _ = chunk_cumprod_householder_fwd_fn(k=k, w1=w1, w2=w2, S=S, BT=BT)
            torch.npu.synchronize()  # 确保 kernel 真正执行完
            prof.step()
        prof.stop()
    
    print("Profiling完成！结果已保存到 ./result_dir_original")


if __name__ == "__main__":
    import functools
    
    # 先运行基本测试
    print("=" * 60)
    print("运行基本功能测试")
    print("=" * 60)
    test_chunk_cumprod_householder_fwd_kernel()
    
    # # 再运行profiling测试
    # print("\n" + "=" * 60)
    # print("运行Profiling测试")
    # print("=" * 60)
    # profile_chunk_cumprod_householder_fwd_kernel()
