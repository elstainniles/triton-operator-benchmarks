from typing import Optional
import torch
import torch_npu
import triton
import triton.language as tl


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
    

# Copy kernel definition from original file
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
    ],
    key=['BT'],
)
@triton.jit(do_not_specialize=['T'])
def save_intra_chunk_attn(
    A,
    A_local,
    T,
    BT: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    p_A = tl.make_block_ptr(A + i_bh * T * T, (T, T), (T, 1), (i_t * BT, i_t * BT), (BT, BT), (1, 0))
    p_A_local = tl.make_block_ptr(A_local + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_A_local = tl.load(p_A_local, boundary_check=(0, 1))
    tl.store(p_A, b_A_local.to(p_A.dtype.element_ty), boundary_check=(0, 1))


def test_save_intra_chunk_attn():
    # Config with large shape for performance testing
    B, H, T = 4, 8, 2048
    BT = 32  # chunk size
    
    # Create input tensors on NPU
    device = torch.device('npu:0')
    
    # Load input tensors from pt file
    import os
    load_dir = "/data/PytorchFile/Inference_or_Training_solution/fla_vv_ops_2/save_intra_chunk_attn"
    inputs_dict = torch.load(os.path.join(load_dir, "inputs.pt"))
    
    # Convert loaded tensors to NPU
    A_local = inputs_dict['A_local'].to(device)
    A = inputs_dict['A'].to(device)
    T = inputs_dict['T']
    BT = inputs_dict['BT']
    
    # Execute kernel once
    print("\nExecuting kernel...")
    grid = (triton.cdiv(T, BT), B * H)
    save_intra_chunk_attn[grid](
        A=A,
        A_local=A_local,
        T=T,
        BT=BT,
    )
    
    # Load expected output tensors from pt file
    outputs_dict = torch.load(os.path.join(load_dir, "outputs.pt"))
    A_expected = outputs_dict['A'].to(device)
    
    # Compare outputs with expected values
    print("\nComparing outputs with expected values...")
    validate_cmp("float16", A, A_expected)
    torch.testing.assert_close(A, A_expected, atol=0.008, rtol=0.008)
    print("Output comparison passed!")
    
    # Warmup run
    print("\nWarmup run...")
    save_intra_chunk_attn[grid](
        A=A,
        A_local=A_local,
        T=T,
        BT=BT,
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
            save_intra_chunk_attn[grid](
                A=A,
                A_local=A_local,
                T=T,
                BT=BT,
            )
            torch.npu.synchronize()
            prof.step()
        prof.stop()
    
    print(f"Test passed! A shape: {A.shape}")
    print(f"Profiling results saved to ./result_dir_original")


if __name__ == "__main__":
    test_save_intra_chunk_attn()
