import os
import time
import shutil
import torch
import triton
import triton.language as tl
import torch_npu
import torch.nn.functional as F
import math
from typing import Tuple
from einops import repeat

import numpy as np

os.environ['TRITON_ALL_BLOCKS_PARALLEL'] = '1'

if os.path.exists("./prof_dacs_segsum_dir"):
    shutil.rmtree("./prof_dacs_segsum_dir")


# @triton.autotune(
#     configs=[
#         triton.Config({}, num_stages=s, num_warps=w)
#         for s in [2, 3]
#         for w in [4, 8]
#     ],
#     key=["CHUNK_SIZE"],
# )
@triton.jit
def dacs_segsum_kernel(
        da_ptr,
        da_cs_ptr,
        da_cs_rev_ptr,
        segsum_ptr,
        stride_da_batch, stride_da_head, stride_da_seq,
        stride_da_cs_batch, stride_da_cs_head, stride_da_cs_seq,
        stride_da_cs_rev_batch, stride_da_cs_rev_head, stride_da_cs_rev_seq,
        stride_segsum_batch, stride_segsum_head, stride_segsum_chunk,
        stride_segsum_row, stride_segsum_col,
        SEQLEN: tl.constexpr,
        CHUNK_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_chunk = tl.program_id(2)

    chunk_start = pid_chunk * CHUNK_SIZE
    offs = tl.arange(0, CHUNK_SIZE)
    offs_seq = chunk_start + offs
    mask = offs_seq < SEQLEN

    base_da = pid_batch * stride_da_batch + pid_head * stride_da_head
    da_chunk = tl.load(da_ptr + base_da + offs_seq * stride_da_seq, mask=mask, other=0.0)

    da_cs = tl.cumsum(da_chunk, axis=0)
    da_cs = tl.minimum(da_cs, 0.0)

    da_cs_rev = tl.cumsum(da_chunk, axis=0, reverse=True)
    i = tl.arange(0, CHUNK_SIZE)[:, None]
    j = tl.arange(0, CHUNK_SIZE)[None, :]
    S_mat = (i == j - 1)
    da_cs_rev = tl.sum(tl.where(S_mat, da_cs_rev, 0), axis=1)
    da_cs_rev = tl.minimum(da_cs_rev, 0.0)

    base_da_cs = pid_batch * stride_da_cs_batch + pid_head * stride_da_cs_head
    base_da_cs_rev = pid_batch * stride_da_cs_rev_batch + pid_head * stride_da_cs_rev_head
    tl.store(da_cs_ptr + base_da_cs + offs_seq * stride_da_cs_seq, da_cs, mask=mask)
    tl.store(da_cs_rev_ptr + base_da_cs_rev + offs_seq * stride_da_cs_rev_seq, da_cs_rev, mask=mask)

    broadcasted_indices = tl.zeros_like(offs)
    segsum = tl.load(da_ptr + base_da + offs_seq[:, None] * stride_da_seq + broadcasted_indices[None, :])
    offs_i = offs[:, None]
    offs_j = offs[None, :]
    segsum = tl.where(offs_i > offs_j, segsum, 0.0)
    segsum = tl.cumsum(segsum, axis=0)
    segsum = tl.minimum(segsum, 0.0)

    base_segsum = (pid_batch * stride_segsum_batch +
                   pid_head * stride_segsum_head +
                   pid_chunk * stride_segsum_chunk)
    tl.store(segsum_ptr + base_segsum + offs_i * stride_segsum_row + offs_j * stride_segsum_col, segsum)


def compute_dacs_segsum_triton(da, chunk_size):
    B, H, S = da.shape
    nchunks = (S + chunk_size - 1) // chunk_size

    da_cs = torch.empty_like(da)
    da_cs_rev = torch.empty_like(da)
    segsum = torch.empty(B, H, nchunks, chunk_size, chunk_size, device=da.device, dtype=da.dtype)

    grid = (B, H, nchunks)
    dacs_segsum_kernel[grid](
        da, da_cs, da_cs_rev, segsum,
        da.stride(0), da.stride(1), da.stride(2),
        da_cs.stride(0), da_cs.stride(1), da_cs.stride(2),
        da_cs_rev.stride(0), da_cs_rev.stride(1), da_cs_rev.stride(2),
        segsum.stride(0), segsum.stride(1), segsum.stride(2),
        segsum.stride(3), segsum.stride(4),
        S,
        chunk_size, enable_graph_optimize=False,
    )

    return da_cs, da_cs_rev, segsum


# def compute_dacs_segsum_ref(da, chunk_size):
#     B, H, S = da.shape
#     nchunks = S // chunk_size
#
#     da_reshaped = da.view(B, H, nchunks, chunk_size)
#     da_cs = torch.cumsum(da_reshaped, dim=-1)
#     da_cs_sum = torch.sum(da_reshaped, dim=-1)
#     da_cs_rev = da_cs_sum[..., None] - da_cs
#
#     segsum = repeat(da_reshaped, "... d -> ... d e", e=chunk_size)
#     mask = torch.tril(torch.ones(chunk_size, chunk_size, device=da_cs.device, dtype=bool), diagonal=-1)
#     segsum = segsum.masked_fill(~mask, 0)
#     segsum = torch.cumsum(segsum, dim=-2)
#
#     return da_cs.view(B, H, S), da_cs_rev.view(B, H, S), segsum


def compute_dacs_segsum_ref(da, chunk_size):
    """
    NumPy 实现原版 torch 函数，结果转回原 device 的 torch tensor
    输入: da (torch.Tensor, NPU/CPU/GPU 均可), chunk_size (int)
    输出: 3 个与输入同 device、同 dtype 的 torch.Tensor
    """
    # 保存原始设备信息（NPU/CPU/GPU）和数据类型
    device = da.device
    dtype = da.dtype

    # 1. 把 torch tensor 转到 CPU 转 numpy（不影响原 tensor）
    da_np = da.detach().cpu().numpy()
    B, H, S = da_np.shape
    nchunks = S // chunk_size

    # 2. reshape 分块 (B, H, nchunks, chunk_size)
    da_reshaped_np = da_np.reshape(B, H, nchunks, chunk_size)

    # 3. 最后一维 cumsum
    da_cs_np = np.cumsum(da_reshaped_np, axis=-1)

    # 4. 最后一维求和
    da_cs_sum_np = np.sum(da_reshaped_np, axis=-1, keepdims=False)

    # 5. 反向 cumsum: sum - cumsum
    da_cs_rev_np = da_cs_sum_np[..., np.newaxis] - da_cs_np

    # 6. 构造 segsum：repeat 扩展维度
    segsum_np = np.repeat(da_reshaped_np[..., np.newaxis], chunk_size, axis=-1)

    # 7. 构造下三角 mask（对角线-1以下为True）
    mask_np = np.tril(np.ones((chunk_size, chunk_size), dtype=bool), k=-1)
    segsum_np = np.where(mask_np, segsum_np, 0.0)  # 非mask位置填0

    # 8. 倒数第二维 cumsum
    segsum_np = np.cumsum(segsum_np, axis=-2)

    # 9. reshape 回原始长度 S
    da_cs_out = da_cs_np.reshape(B, H, S)
    da_cs_rev_out = da_cs_rev_np.reshape(B, H, S)

    # 10. 转回 torch tensor + 放回原设备（NPU）
    da_cs = torch.from_numpy(da_cs_out).to(device=device, dtype=dtype)
    da_cs_rev = torch.from_numpy(da_cs_rev_out).to(device=device, dtype=dtype)
    segsum = torch.from_numpy(segsum_np).to(device=device, dtype=dtype)

    return da_cs, da_cs_rev, segsum

# def generate_data(B, H, S, chunk_size):
#     da = -F.softplus(-3.0 + torch.randn(B, H, S, device='npu', dtype=torch.float))
#     return da

def generate_data(B, H, S, chunk_size):
    """
    NumPy 复现原版 generate_data，结果返回 NPU 上的 torch.float Tensor
    输入: B, H, S, chunk_size (chunk_size 无实际计算作用，保持接口兼容)
    输出: da (torch.Tensor, device='npu', dtype=torch.float)
    """
    # 1. NumPy 生成标准正态分布随机数 (和 torch.randn 完全等价)
    np_rand = np.random.randn(B, H, S).astype(np.float32)

    # 2. 严格复刻原公式：-3.0 + 随机数 → softplus → 取负
    x = np_rand - 3.0  # 对应 -3.0 + torch.randn(...)
    softplus_np = np.log(1 + np.exp(x))  # NumPy 实现 softplus
    da_np = -softplus_np  # 对应 -F.softplus(...)

    # 3. 转回 torch Tensor 并放到 NPU，dtype 保持 torch.float
    da = torch.from_numpy(da_np).to(device='npu', dtype=torch.float)

    return da

SHAPES = [
    (16, 32, 2048, 16),
]


def test_main():
    total_start = time.time()

    for idx, (B, H, S, chunk_size) in enumerate(SHAPES):
        shape_start = time.time()
        print(f"\n{'='*60}")
        print(f"[{idx+1}/{len(SHAPES)}] Testing shape: B={B}, H={H}, S={S}, chunk_size={chunk_size}")
        print(f"{'='*60}")

        da = generate_data(B, H, S, chunk_size)

        da_cs_triton, da_cs_rev_triton, segsum_triton = compute_dacs_segsum_triton(da, chunk_size)
        da_cs_ref, da_cs_rev_ref, segsum_ref = compute_dacs_segsum_ref(da, chunk_size)

        max_diff_cs = (da_cs_ref - da_cs_triton).abs().max().item()
        max_diff_cs_rev = (da_cs_rev_ref - da_cs_rev_triton).abs().max().item()
        max_diff_segsum = (segsum_ref - segsum_triton).abs().max().item()
        print(da_cs_ref.shape, da_cs_rev_ref.shape, segsum_ref.shape)
        assert max_diff_cs < 1e-4, f"da_cs accuracy failed: max_diff={max_diff_cs:.2e}"
        assert max_diff_cs_rev < 1e-4, f"da_cs_rev accuracy failed: max_diff={max_diff_cs_rev:.2e}"
        assert max_diff_segsum < 1e-4, f"segsum accuracy failed: max_diff={max_diff_segsum:.2e}"
        print(f"====accuracy ok (cs={max_diff_cs:.2e}, cs_rev={max_diff_cs_rev:.2e}, segsum={max_diff_segsum:.2e})===")

        for _ in range(5):
            compute_dacs_segsum_triton(da, chunk_size)
            torch_npu.npu.synchronize()

        prof_dir = f"./prof_dacs_segsum_dir/shape_{B}_{H}_{S}_{chunk_size}"

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
                wait=1, warmup=1, active=20, repeat=1, skip_first=1
            ),
            experimental_config=experimental_config,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(prof_dir)
        ) as prof:
            for i in range(40):
                compute_dacs_segsum_triton(da, chunk_size)
                torch_npu.npu.synchronize()
                prof.step()
            prof.stop()

        shape_elapsed = time.time() - shape_start
        print(f"Profile saved to: {prof_dir} (took {shape_elapsed:.1f}s)")

        del da, da_cs_triton, da_cs_rev_triton, segsum_triton, da_cs_ref, da_cs_rev_ref, segsum_ref

    total_elapsed = time.time() - total_start
    print(f"\n\nAll profiling completed in {total_elapsed:.1f}s!")
    print("Results are in ./prof_dacs_segsum_dir/shape_* directories")


if __name__ == "__main__":
    test_main()