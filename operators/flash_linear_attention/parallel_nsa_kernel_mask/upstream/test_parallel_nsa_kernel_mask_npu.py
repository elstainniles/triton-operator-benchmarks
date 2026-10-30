# Test case for parallel_nsa_kernel_mask
# Extracted from fla/ops/nsa/parallel.py

from typing import Optional
import torch
import torch_npu
import triton
import triton.language as tl
import os


def validate_cmp(dtype, y_cal, y_ref, overflow_mode: Optional[str] = None, device_type: Optional[str] = None):
    if device_type is not None:
        target_device = torch.device(device_type)
        y_cal = y_cal.to(target_device)
        y_ref = y_ref.to(target_device)
    else:
        y_cal = y_cal.npu()
        y_ref = y_ref.npu()
    if overflow_mode == "saturate":
        if dtype in ['float32', 'float16']:
            min_value = -torch.finfo(dtype).min
            max_value = torch.finfo(dtype).max
        elif dtype in ['int32', 'int16', 'int8']:
            min_value = torch.iinfo(dtype).min
            max_value = torch.iinfo(dtype).max
        elif dtype == 'bool':
            min_value = 0
            max_value = 1
        else:
            raise ValueError('Invalid parameter "dtype" is found : {}'.format(dtype))
        y_ref = torch.clamp(y_ref, min=min_value, max=max_value)
    if dtype == 'float16':
        torch.testing.assert_close(y_ref, y_cal, rtol=1e-03, atol=1e-03, equal_nan=True)
    elif dtype == 'bfloat16':
        torch.testing.assert_close(y_ref.to(torch.float32), y_cal.to(torch.float32), rtol=5e-03, atol=5e-03,
                                   equal_nan=True)
    elif dtype == 'float32':
        torch.testing.assert_close(y_ref, y_cal, rtol=1e-05, atol=1e-05, equal_nan=True)
    elif dtype == 'int32' or dtype == 'int64' or dtype == 'int16' or dtype == 'int8':
        assert torch.equal(y_cal, y_ref)
    elif dtype == 'bool':
        assert torch.equal(y_cal, y_ref)
    else:
        raise ValueError('Invalid parameter \"dtype\" is found : {}'.format(dtype))

# ============== Kernel definition ==============
@triton.heuristics({
    'USE_BLOCK_COUNTS': lambda args: isinstance(args['block_counts'], torch.Tensor),
})
@triton.jit(do_not_specialize=['T'])
def parallel_nsa_kernel_mask(
    block_indices,
    block_counts,
    block_mask,
    T,
    H: tl.constexpr,
    S: tl.constexpr,
    BS: tl.constexpr,
    NS: tl.constexpr,
    USE_BLOCK_COUNTS: tl.constexpr,
):
    i_t, i_b, i_hs = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h, i_s = i_hs // S, i_hs % S

    b_i = tl.load(block_indices + i_b * T * H * S + i_t * H * S + i_h * S + i_s)
    if USE_BLOCK_COUNTS:
        b_m = b_i * BS <= i_t and i_s < tl.load(block_counts + i_b * T * H + i_t * H + i_h)
    else:
        b_m = b_i * BS <= i_t

    if b_i < NS and b_i >= 0:
        tl.store(block_mask + i_b * T * H * NS + i_t * H * NS + i_h * NS + b_i, b_m.to(block_mask.dtype.element_ty))


# ============== Wrapper function ==============
def parallel_nsa_block_mask(
    block_indices: torch.LongTensor,
    block_counts: torch.LongTensor | int,
    cu_seqlens: torch.LongTensor,
    block_size: int,
):
    B, T, H, S = block_indices.shape
    BS = block_size
    if cu_seqlens is not None:
        # For varlen, we need to compute NS from max sequence length
        # Simplified version for testing
        NS = triton.cdiv(T, BS)
    else:
        NS = triton.cdiv(T, BS)
    block_mask = torch.zeros(B, T, H, NS, dtype=torch.bool, device=block_indices.device)

    parallel_nsa_kernel_mask[(T, B, H*S)](
        block_indices=block_indices,
        block_counts=block_counts,
        block_mask=block_mask,
        T=T,
        H=H,
        S=S,
        BS=BS,
        NS=NS,
    )
    return block_mask


# ============== Test configuration ==============
def get_test_config():
    """Generate test configuration with large shapes"""
    # Large shape configuration for performance testing
    # Note: NPU has a limit on grid dimensions (max 65535 total blocks)
    # Grid is (T, B, H*S), so we need to ensure T * B * H * S <= 65535
    B = 1  # batch size (reduced to fit grid limit)
    T = 512  # sequence length (reduced to fit grid limit)
    H = 4  # number of heads (reduced to fit grid limit)
    block_size = 64
    block_counts = 8  # number of selected blocks (reduced to fit grid limit)
    
    return {
        'B': B,
        'T': T,
        'H': H,
        'block_size': block_size,
        'block_counts': block_counts,
    }


def test_parallel_nsa_kernel_mask():
    """Test function for parallel_nsa_kernel_mask"""
    config = get_test_config()
    B, T, H = config['B'], config['T'], config['H']
    block_size = config['block_size']
    block_counts = config['block_counts']
    
    # Create input tensors on NPU
    device = torch.device('npu:0')
    
    # Load input tensors from pt file
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_vv_ops_2/parallel_nsa_kernel_mask'
    block_indices = torch.load(os.path.join(load_dir, 'block_indices.pt')).to(device)
    print(f"Input tensors loaded from {load_dir}")
    
    print(f"Test configuration:")
    print(f"  Batch size (B): {B}")
    print(f"  Sequence length (T): {T}")
    print(f"  Number of heads (H): {H}")
    print(f"  Block size: {block_size}")
    print(f"  Block counts: {block_counts}")
    print(f"  Block indices shape: {block_indices.shape}")
    
    # Run the kernel
    block_mask = parallel_nsa_block_mask(
        block_indices=block_indices,
        block_counts=block_counts,
        cu_seqlens=None,
        block_size=block_size,
    )
    
    # Load expected output tensors from pt file
    block_mask_expected = torch.load(os.path.join(load_dir, 'block_mask.pt')).to(device)
    print(f"Expected output tensors loaded from {load_dir}")
    
    # Compare outputs with expected values
    print(f"\nComparing outputs with expected values...")
    validate_cmp("int32", block_mask, block_mask_expected)
    print(f"Output comparison passed!")
    
    print(f"\nOutput block_mask shape: {block_mask.shape}")
    print(f"Output dtype: {block_mask.dtype}")
    print(f"Sample block mask values: {block_mask[0, 0, 0, :5]}")
    print(f"\nTest passed!")
    
    return block_mask


def profile_parallel_nsa_kernel_mask():
    """Profiling function for parallel_nsa_kernel_mask"""
    config = get_test_config()
    B, T, H = config['B'], config['T'], config['H']
    block_size = config['block_size']
    block_counts = config['block_counts']
    
    # Create input tensors on NPU
    device = torch.device('npu:0')
    
    # Load input tensors from pt file
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_vv_ops_2/parallel_nsa_kernel_mask'
    block_indices = torch.load(os.path.join(load_dir, 'block_indices.pt')).to(device)
    
    print(f"\nProfiling parallel_nsa_kernel_mask...")
    print(f"  B={B}, T={T}, H={H}")
    
    # Warmup run
    print("\nWarming up...")
    for _ in range(5):
        _ = parallel_nsa_block_mask(
            block_indices=block_indices,
            block_counts=block_counts,
            cu_seqlens=None,
            block_size=block_size,
        )
        torch.npu.synchronize()
    
    # Profiling
    print("\nStarting profiling...")
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
            _ = parallel_nsa_block_mask(
                block_indices=block_indices,
                block_counts=block_counts,
                cu_seqlens=None,
                block_size=block_size,
            )
            torch.npu.synchronize()
            prof.step()
        prof.stop()
    
    print("\nProfiling completed!")
    print("Results saved to: ./result_dir_original")


if __name__ == "__main__":
    test_parallel_nsa_kernel_mask()
    profile_parallel_nsa_kernel_mask()
