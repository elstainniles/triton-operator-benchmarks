import os
import math
import time
import shutil
import torch
import triton
import triton.language as tl

os.environ['TRITON_ALL_BLOCKS_PARALLEL'] = '1'

PROF_DIR = "./prof_bwd_dadt_cumsum_fused_dir"
if os.path.exists(PROF_DIR):
    shutil.rmtree(PROF_DIR)


def to_numpy(x: torch.Tensor):
    return x.detach().cpu().numpy()


def get_num_vector_cores():
    device = torch.npu.current_device()
    device_properties = triton.runtime.driver.active.utils.get_device_properties(device)
    return device_properties.get('num_vectorcore', 56)


# ============================================================================
# Triton kernel: bwd_dadt_cumsum_fused_kernel (dense)
# ============================================================================

@triton.autotune(
    configs=[
        triton.Config({}, num_stages=s, num_warps=w)
        for s in [1, 2, 3]
        for w in [4, 8]
    ],
    key=["CHUNK_SIZE"],
    restore_value=["ddt_out_ptr"],
)
@triton.jit
def bwd_dadt_cumsum_fused_kernel(
        ddA_cs_ptr,
        ddA_cs_rev_ptr,
        dA_cs_ptr,
        dA_cs_rev_ptr,
        ddt_out_ptr,
        stride_batch,
        stride_head,
        stride_seq,
        B: tl.constexpr,
        H: tl.constexpr,
        S: tl.constexpr,
        CHUNK_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_chunk = tl.program_id(2)

    chunk_start = pid_chunk * CHUNK_SIZE
    offs_seq = chunk_start + tl.arange(0, CHUNK_SIZE)
    mask = offs_seq < S

    base_offset = pid_batch * stride_batch + pid_head * stride_head

    ddA_cs = tl.load(ddA_cs_ptr + base_offset + offs_seq * stride_seq, mask=mask, other=0.0)
    ddA_cs_rev = tl.load(ddA_cs_rev_ptr + base_offset + offs_seq * stride_seq, mask=mask, other=0.0)
    dA_cs = tl.load(dA_cs_ptr + base_offset + offs_seq * stride_seq, mask=mask, other=0.0)
    dA_cs_rev = tl.load(dA_cs_rev_ptr + base_offset + offs_seq * stride_seq, mask=mask, other=0.0)

    scaled_ddA_cs = tl.exp(dA_cs) * ddA_cs
    ddt_cs = tl.cumsum(scaled_ddA_cs, axis=0, reverse=True)

    scaled_ddA_cs_rev = tl.exp(dA_cs_rev) * ddA_cs_rev
    ddt_cs_rev_inclusive = tl.cumsum(scaled_ddA_cs_rev, axis=0)

    i = tl.arange(0, CHUNK_SIZE)[:, None]
    j = tl.arange(0, CHUNK_SIZE)[None, :]
    S_mat = (i == j + 1)
    ddt_cs_rev_exclusive = tl.sum(tl.where(S_mat, ddt_cs_rev_inclusive, 0), axis=1)

    ddt_total = ddt_cs + ddt_cs_rev_exclusive
    tl.store(ddt_out_ptr + base_offset + offs_seq * stride_seq, ddt_total, mask=mask)


# ============================================================================
# Triton kernel wrapper
# ============================================================================

def bwd_dadt_cumsum_fused_triton(
        ddA_cs: torch.Tensor,
        ddA_cs_rev: torch.Tensor,
        dA_cs: torch.Tensor,
        dA_cs_rev: torch.Tensor,
        chunk_size: int,
) -> torch.Tensor:
    B, H, S = ddA_cs.shape
    nchunks = S // chunk_size
    assert S % chunk_size == 0, f"S={S} must be divisible by chunk_size={chunk_size}"

    ddt_out = torch.zeros(B, H, S, device=ddA_cs.device, dtype=torch.float32)
    grid = (B, H, nchunks)
    bwd_dadt_cumsum_fused_kernel[grid](
        ddA_cs, ddA_cs_rev, dA_cs, dA_cs_rev, ddt_out,
        ddA_cs.stride(0), ddA_cs.stride(1), ddA_cs.stride(2),
        B, H, S,
        CHUNK_SIZE=chunk_size, enable_graph_optimize=False,
    )
    return ddt_out


# ============================================================================
# PyTorch reference implementations
# ============================================================================

def bwd_ddt_from_ddA_cs_ref(
        ddA_cs: torch.Tensor,
        dA_cs: torch.Tensor,
        chunk_size: int,
) -> torch.Tensor:
    B, H, S = ddA_cs.shape
    nchunks = S // chunk_size
    ddA_cs = torch.exp(dA_cs) * ddA_cs
    ddA_cs = ddA_cs.view(B, H, nchunks, chunk_size)
    ddA = torch.flip(torch.cumsum(torch.flip(ddA_cs, dims=[-1]), dim=-1), dims=[-1])
    ddt = ddA * (-math.log2(math.e))
    return ddt.reshape(B, H, nchunks * chunk_size)


def bwd_ddt_from_ddA_cs_rev_ref(
        ddA_cs_rev: torch.Tensor,
        dA_cs_rev: torch.Tensor,
        chunk_size: int,
) -> torch.Tensor:
    B, H, S = ddA_cs_rev.shape
    nchunks = S // chunk_size
    ddA_cs_rev = torch.exp(dA_cs_rev) * ddA_cs_rev
    ddA_cs_rev = ddA_cs_rev.view(B, H, nchunks, chunk_size)
    ddA = torch.cumsum(ddA_cs_rev, dim=-1)
    ddA = torch.cat([torch.zeros_like(ddA[..., :1]), ddA[..., :-1]], dim=-1)
    ddt = ddA * (-math.log2(math.e))
    return ddt.reshape(B, H, nchunks * chunk_size)


def bwd_dadt_cumsum_fused_ref(
        ddA_cs: torch.Tensor,
        ddA_cs_rev: torch.Tensor,
        dA_cs: torch.Tensor,
        dA_cs_rev: torch.Tensor,
        chunk_size: int,
) -> torch.Tensor:
    ddt1 = bwd_ddt_from_ddA_cs_ref(ddA_cs, dA_cs, chunk_size)
    ddt2 = bwd_ddt_from_ddA_cs_rev_ref(ddA_cs_rev, dA_cs_rev, chunk_size)
    return ddt1 + ddt2


# ============================================================================
# Data generation
# ============================================================================

def generate_data(B: int, H: int, S: int, chunk_size: int, scale: float = 0.1):
    torch.manual_seed(42)
    ddA_cs = torch.randn(B, H, S, device='npu', dtype=torch.float32)
    ddA_cs_rev = torch.randn(B, H, S, device='npu', dtype=torch.float32)
    dA_cs = torch.randn(B, H, S, device='npu', dtype=torch.float32) * scale
    dA_cs_rev = torch.randn(B, H, S, device='npu', dtype=torch.float32) * scale
    return ddA_cs, ddA_cs_rev, dA_cs, dA_cs_rev


# ============================================================================
# Test shapes
# ============================================================================

SHAPES = [
    (16, 32, 2048, 16),
]


def test_main():
    import torch_npu

    total_start = time.time()

    for idx, (B, H, S, chunk_size) in enumerate(SHAPES):
        shape_start = time.time()
        print(f"\n{'='*60}")
        print(f"[{idx+1}/{len(SHAPES)}] Testing shape: B={B}, H={H}, S={S}, chunk_size={chunk_size}")
        print(f"{'='*60}")

        ddA_cs, ddA_cs_rev, dA_cs, dA_cs_rev = generate_data(B, H, S, chunk_size)

        # ---- Precision test ----
        result = bwd_dadt_cumsum_fused_triton(ddA_cs, ddA_cs_rev, dA_cs, dA_cs_rev, chunk_size)*(-math.log2(math.e))
        golden = bwd_dadt_cumsum_fused_ref(ddA_cs, ddA_cs_rev, dA_cs, dA_cs_rev, chunk_size)

        max_diff = (result - golden).abs().max().item()
        mean_diff = (result - golden).abs().mean().item()
        rel_diff = ((result - golden).abs() / (golden.abs() + 1e-8)).max().item()

        print(f"  Max abs diff:  {max_diff:.2e}")
        print(f"  Mean abs diff: {mean_diff:.2e}")
        print(f"  Max rel diff:  {rel_diff:.2e}")

        assert max_diff < 1e-4, f"Max diff: {max_diff:.2e} (threshold: 1e-4)"
        if max_diff < 1e-4:
            print("  ====accuracy ok===")
        else:
            print(f"  ====accuracy FAIL (max_diff={max_diff:.2e} > 1e-4)====")

        # ---- Warmup ----
        for _ in range(5):
            out = bwd_dadt_cumsum_fused_triton(ddA_cs, ddA_cs_rev, dA_cs, dA_cs_rev, chunk_size)
            torch_npu.npu.synchronize()

        # ---- Performance profiling ----
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
                out = bwd_dadt_cumsum_fused_triton(ddA_cs, ddA_cs_rev, dA_cs, dA_cs_rev, chunk_size)
                torch_npu.npu.synchronize()
                prof.step()
            prof.stop()

        shape_elapsed = time.time() - shape_start
        print(f"  Profile saved to: {prof_dir} (took {shape_elapsed:.1f}s)")

        del ddA_cs, ddA_cs_rev, dA_cs, dA_cs_rev, out, result, golden

    total_elapsed = time.time() - total_start
    print(f"\n\nAll profiling completed in {total_elapsed:.1f}s!")
    print(f"Results are in {PROF_DIR}/shape_* directories")


if __name__ == "__main__":
    test_main()