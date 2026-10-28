import torch
import math
import sys
import os
import triton
import triton.backends.ascend.runtime
import triton.language as tl
import torch_npu
import shutil
NEG_LOG2E = -1.4426950408889634
os.environ["TRITON_ALL_BLOCKS_PARALLEL"] = "1"

PROF_DIR = "./prof_bwd_segsum_dadt_kernel_dir"
if os.path.exists(PROF_DIR):
    shutil.rmtree(PROF_DIR)

@triton.autotune(
    configs=[
        triton.Config({}, num_stages=s, num_warps=w)
        for s in [2, 3]
        for w in [4, 8]
    ],
    key=["CHUNK_SIZE"],
    restore_value=["ddt_out_ptr"],
)
@triton.jit
def bwd_segsum_dadt_kernel(
        dSSdA_ptr,
        SSdA_cs_ptr,
        ddt_out_ptr,
        stride_dSSdA_batch,
        stride_dSSdA_head,
        stride_dSSdA_chunk,
        stride_dSSdA_row,
        stride_dSSdA_col,
        stride_SSdA_batch,
        stride_SSdA_head,
        stride_SSdA_chunk,
        stride_SSdA_row,
        stride_SSdA_col,
        stride_ddt_batch,
        stride_ddt_head,
        stride_ddt_seq,
        B: tl.constexpr,
        H: tl.constexpr,
        nchunks: tl.constexpr,
        C: tl.constexpr,
        CHUNK_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_chunk = tl.program_id(2)

    chunk_start = pid_chunk * CHUNK_SIZE
    offs_c = tl.arange(0, CHUNK_SIZE)
    offs_seq = chunk_start + offs_c

    dSSdA_offset = dSSdA_ptr + (pid_batch * stride_dSSdA_batch +
                                pid_head * stride_dSSdA_head +
                                pid_chunk * stride_dSSdA_chunk)
    SSdA_offset = SSdA_cs_ptr + (pid_batch * stride_SSdA_batch +
                                 pid_head * stride_SSdA_head +
                                 pid_chunk * stride_SSdA_chunk)
    ddt_ptrs = ddt_out_ptr + (pid_batch * stride_ddt_batch +
                              pid_head * stride_ddt_head +
                              offs_seq * stride_ddt_seq)

    dSSdA_block = tl.load(dSSdA_offset + offs_c[:, None] * stride_dSSdA_col + offs_c[None, :] * stride_dSSdA_row)
    SSdA_block = tl.load(SSdA_offset + offs_c[:, None] * stride_SSdA_row + offs_c[None, :] * stride_SSdA_col)

    dSSdA_block = dSSdA_block * tl.exp(SSdA_block)
    dSSdA_block = tl.cumsum(dSSdA_block, axis=0, reverse=True)

    offs_i = tl.arange(0, CHUNK_SIZE)[:, None]
    offs_j = tl.arange(0, CHUNK_SIZE)[None, :]
    SS_mask = offs_i > offs_j
    dSSdA = tl.where(SS_mask, dSSdA_block, 0.0)

    ddt_chunk = tl.load(ddt_ptrs)
    ddt_chunk += tl.sum(dSSdA, axis=1)
    tl.store(ddt_ptrs, ddt_chunk)


def bwd_segsum_dadt_triton(dSSdA, SSdA, chunk_size):
    B, H, nchunks, C, C_ = dSSdA.shape # torch.Size([16, 32, 128, 16, 16])
    S = nchunks * chunk_size # 2048
    ddt_out = torch.zeros(B, H, S, device=dSSdA.device, dtype=torch.float32) # 16 32 2048
    grid = (B, H, nchunks) # (16 32 128)

    bwd_segsum_dadt_kernel[grid](
        dSSdA, SSdA, ddt_out,
        dSSdA.stride(0), dSSdA.stride(1), dSSdA.stride(2), # 1048576, 32768, 256
        dSSdA.stride(3), dSSdA.stride(4), # 16 1
        SSdA.stride(0), SSdA.stride(1), SSdA.stride(2), #1048576 32768 256
        SSdA.stride(3), SSdA.stride(4), # 16 1
        ddt_out.stride(0), ddt_out.stride(1), ddt_out.stride(2), # 65536 2048 1
        B, H, nchunks, chunk_size,
        CHUNK_SIZE=chunk_size, enable_graph_optimize=False,
    )
    return ddt_out


def bwd_segsum_ddt_from_dSSdA_ref(dSSdA, dA_cs, chunk_size):
    B, H, nchunks, C, C_ = dSSdA.shape
    dA_cs_chunk = dA_cs.view(B, H, nchunks, C)
    dSSdA = dSSdA.permute([0, 1, 2, 4, 3])
    seg = dA_cs_chunk[..., :, None] - dA_cs_chunk[..., None, :]
    dSSdA = dSSdA * torch.exp(seg)
    ddA = torch.flip(torch.cumsum(torch.flip(dSSdA, dims=[-2]), dim=-2), dims=[-2])
    mask = torch.tril(torch.ones(C, C, device=dSSdA.device, dtype=dSSdA.dtype), -1)
    ddA = ddA * mask
    ddA = ddA.sum(-1)
    ddt = ddA * NEG_LOG2E
    return ddt.reshape(B, H, nchunks * C)


def verify_correctness():
    print("=" * 70)
    print("Precision Verification: bwd_segsum_dadt_kernel (NPU Triton vs PyTorch ref)")
    print("=" * 70)

    torch.manual_seed(42)
    B, H, S = 16, 32, 2048
    chunk_size = 16
    nchunks = S // chunk_size
    C = chunk_size

    dSSdA = torch.randn(B, H, nchunks, C, C, device='npu', dtype=torch.float32)
    dA_cs = torch.randn(B, H, S, device='npu', dtype=torch.float32) * 0.1
    dA_cs_reshape = dA_cs.view(B, H, nchunks, chunk_size)
    SSdA = dA_cs_reshape[:, :, :, :, None] - dA_cs_reshape[:, :, :, None, :]

    ddt_triton = bwd_segsum_dadt_triton(dSSdA, SSdA, chunk_size) * NEG_LOG2E
    ddt_ref = bwd_segsum_ddt_from_dSSdA_ref(dSSdA.clone(), dA_cs.clone(), chunk_size)

    max_diff = (ddt_ref - ddt_triton).abs().max().item()
    mean_diff = (ddt_ref - ddt_triton).abs().mean().item()
    rel_diff = ((ddt_ref - ddt_triton).abs() / (ddt_ref.abs() + 1e-8)).max().item()

    print(f"  Shape: dSSdA={dSSdA.shape}, SSdA={SSdA.shape}, ddt_out={ddt_triton.shape}")
    print(f"  Max absolute difference:  {max_diff:.2e}")
    print(f"  Mean absolute difference: {mean_diff:.2e}")
    print(f"  Max relative difference:  {rel_diff:.2e}")
    passed = max_diff < 1e-4
    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    print()

    prof_dir = f"{PROF_DIR}/shape_{B}_{H}_{S}_{chunk_size}"
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
            ddt_triton = bwd_segsum_dadt_triton(dSSdA, SSdA, chunk_size) * NEG_LOG2E
            ddt_ref = bwd_segsum_ddt_from_dSSdA_ref(dSSdA.clone(), dA_cs.clone(), chunk_size)
            torch_npu.npu.synchronize()
            prof.step()
        prof.stop()


    return passed

def profile_with_torch_profiler():
    print("=" * 70)
    print("Performance Profiling: torch_npu.profiler (NPU)")
    print("=" * 70)

    torch.manual_seed(42)
    B, H, S = 16, 32, 2048
    chunk_size = 16
    nchunks = S // chunk_size
    C = chunk_size

    dSSdA = torch.randn(B, H, nchunks, C, C, device='npu', dtype=torch.float32)
    dA_cs = torch.randn(B, H, S, device='npu', dtype=torch.float32) * 0.1
    dA_cs_reshape = dA_cs.view(B, H, nchunks, chunk_size)
    SSdA = dA_cs_reshape[:, :, :, :, None] - dA_cs_reshape[:, :, :, None, :]

    for _ in range(5):
        _ = bwd_segsum_dadt_triton(dSSdA, SSdA, chunk_size)
    torch.npu.synchronize()

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiler_output")
    os.makedirs(output_dir, exist_ok=True)

    # ② 使用 torch_npu.profiler 替代 torch.profiler
    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_modules=True,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(output_dir),
    ) as prof:
        for _ in range(10):
            ddt_out = bwd_segsum_dadt_triton(dSSdA, SSdA, chunk_size)
            torch.npu.synchronize()
            prof.step()          # ③ 标记 step

    print(f"  Profiler output saved to: {output_dir}")
    print()
    print("  Top NPU kernels by self NPU time:")
    print("-" * 70)
    for evt in prof.key_averages().sort_by("self_npu_time_total")[:10]:
        print(f"    {evt.key:50s}  NPU_self={evt.self_npu_time_total / 1e3:8.3f}ms  "
              f"NPU_total={evt.npu_time_total / 1e3:8.3f}ms  "
              f"Calls={evt.count}")
    print()
    print("  Top CPU ops by self CPU time:")
    print("-" * 70)
    for evt in prof.key_averages().sort_by("self_cpu_time_total")[:5]:
        print(f"    {evt.key:50s}  CPU_self={evt.self_cpu_time_total / 1e3:8.3f}ms  "
              f"Calls={evt.count}")
    print()

    return output_dir

def benchmark_latency():
    print("=" * 70)
    print("Latency Benchmark: bwd_segsum_dadt_kernel (NPU)")
    print("=" * 70)

    from triton.testing import do_bench

    torch.manual_seed(42)
    B, H, S = 16, 32, 2048
    chunk_size = 16
    nchunks = S // chunk_size
    C = chunk_size

    dSSdA = torch.randn(B, H, nchunks, C, C, device='npu', dtype=torch.float32)
    dA_cs = torch.randn(B, H, S, device='npu', dtype=torch.float32) * 0.1
    dA_cs_reshape = dA_cs.view(B, H, nchunks, chunk_size)
    SSdA = dA_cs_reshape[:, :, :, :, None] - dA_cs_reshape[:, :, :, None, :]

    triton_time = do_bench(lambda: bwd_segsum_dadt_triton(dSSdA, SSdA, chunk_size), warmup=25, rep=100)
    ref_time = do_bench(lambda: bwd_segsum_ddt_from_dSSdA_ref(dSSdA, dA_cs, chunk_size), warmup=25, rep=100)

    print(f"  Configuration: B={B}, H={H}, S={S}, chunk_size={chunk_size}")
    print(f"  PyTorch reference: {ref_time:.3f} ms")
    print(f"  Triton kernel:     {triton_time:.3f} ms")
    print(f"  Speedup:           {ref_time / triton_time:.2f}x")
    print()

    return ref_time, triton_time


def test_main():
    passed = verify_correctness()
    # profile_with_torch_profiler()
    if not passed:
        print("Precision verification FAILED, skipping profiling.")
        sys.exit(1)


if __name__ == "__main__":
    test_main()
