# Test case for chunk_hgrn_bwd_kernel_o
# Extracted from: fla/ops/hgrn/chunk.py

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

# ============================================================================
# Dependencies from fla.ops.utils.op
# ============================================================================
@triton.jit
def exp(x): 
    return tl.exp(x.to(tl.float32))

# ============================================================================
# Dependencies from fla.utils
# ============================================================================
import functools
import inspect
from packaging import version

SUPPORTS_AUTOTUNE_CACHE = "cache_results" in inspect.signature(triton.autotune).parameters
FLA_CACHE_RESULTS = os.getenv('FLA_CACHE_RESULTS', '1') == '1'
autotune_cache_kwargs = {"cache_results": FLA_CACHE_RESULTS} if SUPPORTS_AUTOTUNE_CACHE else {}

import contextlib
def input_guard(
    fn=None,
    *,
    no_guard_contiguous=False,
):
    def decorator(fn):
        sig = inspect.signature(fn)
        param_names = list(sig.parameters.keys())

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            skip_params = set()
            if isinstance(no_guard_contiguous, list):
                skip_params = set(no_guard_contiguous)

            processed_args = []
            for i, arg in enumerate(args):
                if i < len(param_names):
                    param_name = param_names[i]
                else:
                    param_name = f"__arg_{i}"

                if isinstance(arg, torch.Tensor):
                    if no_guard_contiguous is True or param_name in skip_params:
                        processed_args.append(arg)
                    else:
                        processed_args.append(arg.contiguous())
                else:
                    processed_args.append(arg)

            processed_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor):
                    if no_guard_contiguous is True or k in skip_params:
                        processed_kwargs[k] = v
                    else:
                        processed_kwargs[k] = v.contiguous()
                else:
                    processed_kwargs[k] = v

            tensor = None
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    tensor = arg
                    break
            if tensor is None:
                for value in kwargs.values():
                    if isinstance(value, torch.Tensor):
                        tensor = value
                        break

            if tensor is not None:
                ctx = torch.npu.device(tensor.device.index)
            else:
                ctx = contextlib.nullcontext()

            with ctx:
                return fn(*processed_args, **processed_kwargs)

        return wrapper

    if fn is not None:
        return decorator(fn)

    return decorator

# ============================================================================
# Forward Kernel h (needed to generate inputs for backward)
# ============================================================================
@triton.autotune(
    configs=[
        triton.Config({'BD': 32}, num_warps=1),
        triton.Config({'BD': 32}, num_warps=2),
        triton.Config({'BD': 32}, num_warps=4),
        triton.Config({'BD': 32}, num_warps=8),
        triton.Config({'BD': 64}, num_warps=1),
        triton.Config({'BD': 64}, num_warps=2),
        triton.Config({'BD': 64}, num_warps=4),
        triton.Config({'BD': 64}, num_warps=8),
        triton.Config({'BD': 128}, num_warps=1),
        triton.Config({'BD': 128}, num_warps=2),
        triton.Config({'BD': 128}, num_warps=4),
        triton.Config({'BD': 128}, num_warps=8),
    ],
    key=['D'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_hgrn_fwd_kernel_h(
    x,
    g,
    gc,
    o,
    h0,
    T,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
):
    i_d, i_t, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_d = i_d * BD + tl.arange(0, BD)
    mask = o_d < D

    p_x = x + i_b * T * D + i_t * BT * D + o_d
    p_g = g + i_b * T * D + i_t * BT * D + o_d
    p_gc = gc + i_b * T * D + i_t * BT * D + o_d
    p_o = o + i_b * T * D + i_t * BT * D + o_d

    b_h = tl.zeros([BD], dtype=tl.float32)
    b_gc = tl.zeros([BD], dtype=tl.float32)
    if USE_INITIAL_STATE:
        if i_t == 0:
            b_h += tl.load(h0 + i_b * D + o_d, mask=mask, other=0).to(tl.float32)
    for i in range(0, BT):
        mask_t = mask & ((i_t * BT + i) < T)
        b_x = tl.load(p_x, mask=mask_t, other=0).to(tl.float32)
        b_g = tl.load(p_g, mask=mask_t, other=0).to(tl.float32)
        b_h = exp(b_g) * b_h + b_x
        b_gc = b_gc + b_g
        tl.store(p_gc, b_gc.to(p_o.dtype.element_ty), mask=mask_t)
        tl.store(p_o, b_h.to(p_o.dtype.element_ty), mask=mask_t)

        p_x += D
        p_g += D
        p_gc += D
        p_o += D


# ============================================================================
# Backward Kernel h (needed to generate inputs for kernel_o)
# ============================================================================
@triton.autotune(
    configs=[
        triton.Config({'BD': BD}, num_warps=num_warps)
        for BD in [32, 64, 128]
        for num_warps in [1, 2, 4, 8]
    ],
    key=['D'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_hgrn_bwd_kernel_h(
    g,
    gc,
    dx,
    do,
    T,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    i_d, i_t, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_d = i_d * BD + tl.arange(0, BD)
    mask = o_d < D
    BC = min(BT, T - i_t * BT)
    NT = tl.num_programs(1)

    p_g = g + (i_b * T + i_t * BT + BC - 1) * D + o_d
    p_gc = gc + (i_b * T + i_t * BT + BC - 1) * D + o_d
    p_dx = dx + (i_b * T + i_t * BT + BC - 1) * D + o_d
    p_do = do + (i_b * T + i_t * BT + BC - 1) * D + o_d

    if i_t == NT - 1:
        b_gc = tl.zeros([BD], dtype=tl.float32)
    else:
        b_gc = tl.load(g + (i_b * T + i_t * BT + BT) * D + o_d, mask=mask, other=0).to(tl.float32)
    b_dh = tl.zeros([BD], dtype=tl.float32)
    for _ in range(BC - 1, -1, -1):
        tl.store(p_gc, b_gc.to(p_gc.dtype.element_ty), mask=mask)

        b_g = tl.load(p_g, mask=mask, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask, other=0).to(tl.float32)

        b_gc = b_gc + b_g
        b_dh = b_dh + b_do
        b_dx = b_dh
        b_dh = b_dh * exp(b_g)

        tl.store(p_dx, b_dx.to(p_dx.dtype.element_ty), mask=mask)

        p_g -= D
        p_gc -= D
        p_dx -= D
        p_do -= D


# ============================================================================
# Backward Kernel Definition: chunk_hgrn_bwd_kernel_o
# ============================================================================
@triton.jit(do_not_specialize=['T'])
def chunk_hgrn_bwd_kernel_o(
    g,
    gc,
    o,
    dx,
    dg,
    s_b,
    s_t,
    s_d,
    T,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    i_d, i_b = tl.program_id(0), tl.program_id(1)
    o_d = i_d * BD + tl.arange(0, BD)
    mask = o_d < D

    for i_t in range(tl.cdiv(T, BT) - 1, -1, -1):
        p_g = tl.make_block_ptr(g + i_b * s_b, (T, D), (s_t, s_d), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
        p_gc = tl.make_block_ptr(gc + i_b * s_b, (T, D), (s_t, s_d), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
        p_o = tl.make_block_ptr(o + i_b * s_b, (T, D), (s_t, s_d), (i_t * BT - 1, i_d * BD), (BT, BD), (1, 0))
        p_dx = tl.make_block_ptr(dx + i_b * s_b, (T, D), (s_t, s_d), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
        p_dg = tl.make_block_ptr(dg + i_b * s_b, (T, D), (s_t, s_d), (i_t * BT, i_d * BD), (BT, BD), (1, 0))

        # [BD,]
        mask_t = mask & ((i_t + 1) * BT < T)
        b_ht = tl.load(dx + i_b * T * D + (i_t + 1) * BT * D + o_d, mask=mask_t, other=0).to(tl.float32)
        # [BT, BD]
        b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
        b_gc = tl.load(p_gc, boundary_check=(0, 1)).to(tl.float32)
        b_o = tl.load(p_o, boundary_check=(0, 1)).to(tl.float32)
        b_dx = tl.load(p_dx, boundary_check=(0, 1)).to(tl.float32)

        b_dx = b_dx + exp(b_gc) * b_ht[None, :]
        b_dg = b_o * b_dx * exp(b_g)
        tl.store(p_dx, b_dx.to(p_dx.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0, 1))


# ============================================================================
# Call Functions
# ============================================================================
def chunk_hgrn_fwd_h_call(
    x: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor = None,
):
    B, T, D = x.shape
    BT, BD = 128, min(64, triton.next_power_of_2(D))
    
    gc = torch.empty_like(g, dtype=torch.float)
    o = torch.empty_like(x, dtype=torch.float)
    
    def grid(meta): return (triton.cdiv(D, meta['BD']), triton.cdiv(T, meta['BT']), B)
    chunk_hgrn_fwd_kernel_h[grid](
        x, g, gc, o, initial_state,
        T=T, D=D, BT=BT,
        USE_INITIAL_STATE=initial_state is not None,
    )
    
    return o, gc


def chunk_hgrn_bwd_h_call(
    g: torch.Tensor,
    do: torch.Tensor,
):
    B, T, D = do.shape
    BT, BD = 128, min(64, triton.next_power_of_2(D))
    
    gc = torch.empty_like(g, dtype=torch.float)
    dx = torch.empty_like(do, dtype=torch.float)
    
    def grid(meta): return (triton.cdiv(D, meta['BD']), triton.cdiv(T, meta['BT']), B)
    chunk_hgrn_bwd_kernel_h[grid](
        g, gc, dx, do,
        T=T, D=D, BT=BT,
    )
    
    return dx, gc


def chunk_hgrn_bwd_o_call(
    g: torch.Tensor,
    gc: torch.Tensor,
    o: torch.Tensor,
    dx: torch.Tensor,
):
    B, T, D = dx.shape
    BT, BD = 64, min(64, triton.next_power_of_2(D))
    num_warps = 8 if BD == 64 else 4
    
    dg = torch.empty_like(g, dtype=torch.float)
    
    def grid(meta): return (triton.cdiv(D, meta['BD']), B)
    chunk_hgrn_bwd_kernel_o[grid](
        g, gc, o, dx, dg,
        o.stride(-3), o.stride(-2), o.stride(-1),
        T=T, D=D, BT=BT, BD=BD,
        num_warps=num_warps,
    )
    
    return dx, dg


# ============================================================================
# Test Case Configuration
# ============================================================================
# Using moderate shapes for performance testing (reduced to avoid timeout)
def test_chunk_hgrn_fwd_h():
    B, T, D = 2, 1024, 256

    print(f"Test Configuration: B={B}, T={T}, D={D}")
    print(f"Input shape: ({B}, {T}, {D})")

    # Load input tensors from pt files
    load_dir = "/data/PytorchFile/Inference_or_Training_solution/fla_vv_ops_2/chunk_hgrn_bwd_kernel_o"

    device = torch.device("npu:0")
    x = torch.load(os.path.join(load_dir, "x.pt")).to(device)
    g = torch.load(os.path.join(load_dir, "g.pt")).to(device)
    h0 = torch.load(os.path.join(load_dir, "h0.pt")).to(device)
    do = torch.load(os.path.join(load_dir, "do.pt")).to(device)
    o = torch.load(os.path.join(load_dir, "o.pt")).to(device)
    gc = torch.load(os.path.join(load_dir, "gc.pt")).to(device)
    dx = torch.load(os.path.join(load_dir, "dx.pt")).to(device)

    print(f"\nInput tensors loaded from {load_dir}")
    print(f"x shape: {x.shape}, dtype: {x.dtype}")
    print(f"g shape: {g.shape}, dtype: {g.dtype}")
    print(f"h0 shape: {h0.shape}, dtype: {h0.dtype}")
    print(f"do shape: {do.shape}, dtype: {do.dtype}")
    print(f"o shape: {o.shape}, dtype: {o.dtype}")
    print(f"gc shape: {gc.shape}, dtype: {gc.dtype}")
    print(f"dx shape: {dx.shape}, dtype: {dx.dtype}")

    # Execute bwd_o kernel once
    dx_out, dg = chunk_hgrn_bwd_o_call(
        g=g,
        gc=gc,
        o=o,
        dx=dx.clone()  # Clone to avoid in-place modification
    )

    # Load expected outputs and compare
    expected_dx_out = torch.load(os.path.join(load_dir, "dx_out.pt")).to(device)
    expected_dg = torch.load(os.path.join(load_dir, "dg.pt")).to(device)

    print("Comparing outputs with expected values...")
    validate_cmp("float32", dx_out, expected_dx_out)
    validate_cmp("float32", dg, expected_dg)
    print("✓ Output verification passed!")

    # ============================================================================
    # Profiling Test
    # ============================================================================
    print("\n" + "="*80)
    print("Starting Profiling Test for chunk_hgrn_bwd_kernel_o")
    print("="*80)

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        l2_cache=False
    )

    result_dir = "./result_dir_chunk_hgrn_bwd_kernel_o"

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
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(result_dir)
    ) as prof:
        prof.start()
        for i in range(35):
            dx_out, dg = chunk_hgrn_bwd_o_call(
                g=g,
                gc=gc,
                o=o,
                dx=dx.clone()  # Clone to avoid in-place modification
            )
            torch.npu.synchronize()
            prof.step()
        prof.stop()

    print(f"\nProfiling completed!")
    print(f"Results saved to: {result_dir}")
    print(f"\nBackward output shapes:")
    print(f"dx_out shape: {dx_out.shape}")
    print(f"dg shape: {dg.shape}")
    print("\nTest completed successfully!")
