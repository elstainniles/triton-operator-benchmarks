# Test case for fused_recurrent_hgrn_bwd_kernel
# Extracted from: fla/ops/hgrn/fused_recurrent.py

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
# Forward Kernel (needed to generate inputs for backward)
# ============================================================================
@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
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
def fused_recurrent_hgrn_fwd_kernel(
    x,
    g,
    o,
    h0,
    ht,
    cu_seqlens,
    T,
    D: tl.constexpr,
    BD: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_d, i_n = tl.program_id(0), tl.program_id(1)
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T

    o_d = i_d * BD + tl.arange(0, BD)
    mask = o_d < D

    p_x = x + bos * D + o_d
    p_g = g + bos * D + o_d
    p_o = o + bos * D + o_d

    b_h = tl.zeros([BD], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_n * D + o_d
        b_h += tl.load(p_h0, mask=mask, other=0).to(tl.float32)
    for _ in range(0, T):
        b_x = tl.load(p_x, mask=mask, other=0).to(tl.float32)
        b_g = tl.load(p_g, mask=mask, other=0).to(tl.float32)
        b_h = exp(b_g) * b_h + b_x
        tl.store(p_o, b_h.to(p_o.dtype.element_ty), mask=mask)

        p_x += D
        p_g += D
        p_o += D

    if STORE_FINAL_STATE:
        p_ht = ht + i_n * D + o_d
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask)


# ============================================================================
# Backward Kernel Definition: fused_recurrent_hgrn_bwd_kernel
# ============================================================================
@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
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
def fused_recurrent_hgrn_bwd_kernel(
    g,
    o,
    h0,
    dx,
    dg,
    do,
    dht,
    dh0,
    cu_seqlens,
    T,
    D: tl.constexpr,
    BD: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_d, i_n = tl.program_id(0), tl.program_id(1)
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T

    o_d = i_d * BD + tl.arange(0, BD)
    mask = o_d < D

    p_g = g + (bos + T - 1) * D + o_d
    p_o = o + (bos + T - 2) * D + o_d
    p_dx = dx + (bos + T - 1) * D + o_d
    p_dg = dg + (bos + T - 1) * D + o_d
    p_do = do + (bos + T - 1) * D + o_d

    b_dh = tl.zeros([BD], dtype=tl.float32)
    if USE_FINAL_STATE_GRADIENT:
        p_dht = dht + i_n * D + o_d
        b_dh += tl.load(p_dht, mask=mask, other=0).to(tl.float32)

    for i in range(T - 1, -1, -1):
        b_g = tl.load(p_g, mask=mask, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask, other=0).to(tl.float32)
        if i > 0:
            b_o = tl.load(p_o, mask=mask, other=0).to(tl.float32)
        elif USE_INITIAL_STATE:
            b_o = tl.load(h0 + i_n * D + o_d, mask=mask, other=0).to(tl.float32)
        else:
            b_o = tl.zeros([BD], dtype=tl.float32)

        b_dh = b_dh + b_do
        b_dx = b_dh
        b_dh = b_dh * exp(b_g)
        b_dg = b_dh * b_o
        tl.store(p_dx, b_dx.to(p_dx.dtype.element_ty), mask=mask)
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), mask=mask)

        p_g -= D
        p_o -= D
        p_dx -= D
        p_dg -= D
        p_do -= D

    if USE_INITIAL_STATE:
        p_dh0 = dh0 + i_n * D + o_d
        tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), mask=mask)


# ============================================================================
# Call Functions
# ============================================================================
def fused_recurrent_hgrn_fwd(
    x: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, D = x.shape
    N = B if cu_seqlens is None else len(cu_seqlens) - 1

    o = torch.empty_like(x)
    final_state = x.new_empty(N, D) if output_final_state else None

    def grid(meta): return (triton.cdiv(D, meta['BD']), N)
    fused_recurrent_hgrn_fwd_kernel[grid](
        x=x,
        g=g,
        o=o,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        T=T,
        D=D,
    )
    return o, final_state


def fused_recurrent_hgrn_bwd(
    g: torch.Tensor,
    o: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor = None,
    initial_state: torch.Tensor = None,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, D = do.shape
    N = B if cu_seqlens is None else len(cu_seqlens) - 1

    dx = torch.empty_like(o, dtype=torch.float)
    dg = torch.empty_like(g, dtype=torch.float)
    dh0 = torch.empty_like(initial_state, dtype=torch.float) if initial_state is not None else None
    def grid(meta): return (triton.cdiv(D, meta['BD']), N)
    fused_recurrent_hgrn_bwd_kernel[grid](
        g=g,
        o=o,
        h0=initial_state,
        dx=dx,
        dg=dg,
        do=do,
        dht=dht,
        dh0=dh0,
        cu_seqlens=cu_seqlens,
        T=T,
        D=D,
    )
    return dx, dg, dh0


# ============================================================================
# Test Case Configuration
# ============================================================================
# Using large shapes for performance testing
def test_fused_recurrent_hgrn_bwd_kernel_npu():
    B, T, D = 4, 2048, 512

    print(f"Test Configuration: B={B}, T={T}, D={D}")
    print(f"Input shape: ({B}, {T}, {D})")

    # Load inputs from saved pt file
    device = torch.device("npu:0")
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_vv_ops_2/fused_recurrent_hgrn_bwd_kernel'
    inputs_dict = torch.load(os.path.join(load_dir, 'inputs.pt'))

    # Convert inputs to npu
    g = inputs_dict['g'].to(device)
    o = inputs_dict['o'].to(device)
    do = inputs_dict['do'].to(device)
    dht = inputs_dict['dht'].to(device) if inputs_dict['dht'] is not None else None
    h0 = inputs_dict['h0'].to(device)

    print(f"\nInputs loaded from {load_dir}/inputs.pt")
    print(f"g shape: {g.shape}, dtype: {g.dtype}")
    print(f"o shape: {o.shape}, dtype: {o.dtype}")
    print(f"do shape: {do.shape}, dtype: {do.dtype}")
    print(f"dht shape: {dht.shape if dht is not None else 'None'}")
    print(f"h0 shape: {h0.shape}, dtype: {h0.dtype}")

    # Execute backward kernel once for correctness check
    dx, dg, dh0_out = fused_recurrent_hgrn_bwd(
        g=g,
        o=o,
        do=do,
        dht=dht,
        initial_state=h0
    )
    torch.npu.synchronize()

    # Load expected outputs and compare
    expected_outputs = torch.load(os.path.join(load_dir, 'outputs.pt'))

    # Convert expected outputs to npu for comparison
    expected_dx = expected_outputs['dx'].to(device)
    expected_dg = expected_outputs['dg'].to(device)
    expected_dh0 = expected_outputs['dh0'].to(device) if expected_outputs['dh0'] is not None else None

    # Compare outputs with tolerance
    print("\nComparing outputs...")
    validate_cmp("float32", dx, expected_dx)
    torch.testing.assert_close(dx, expected_dx, atol=0.008, rtol=0.008)
    validate_cmp("float32", dg, expected_dg)
    torch.testing.assert_close(dg, expected_dg, atol=0.008, rtol=0.008)
    if expected_dh0 is not None:
        validate_cmp("float32", dh0_out, expected_dh0)
        torch.testing.assert_close(dh0_out, expected_dh0, atol=0.008, rtol=0.008)
    print("All outputs match within tolerance!")

    # ============================================================================
    # Profiling Test
    # ============================================================================
    print("\n" + "="*80)
    print("Starting Profiling Test for Backward Kernel")
    print("="*80)

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        l2_cache=False
    )

    result_dir = "./result_dir_fused_recurrent_hgrn_bwd_kernel"

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
            dx, dg, dh0_out = fused_recurrent_hgrn_bwd(
                g=g,
                o=o,
                do=do,
                dht=dht,
                initial_state=h0
            )
            torch.npu.synchronize()
            prof.step()
        prof.stop()

    print(f"\nProfiling completed!")
    print(f"Results saved to: {result_dir}")
    print(f"\nBackward output shapes:")
    print(f"dx shape: {dx.shape}")
    print(f"dg shape: {dg.shape}")
    print(f"dh0 shape: {dh0_out.shape if dh0_out is not None else 'None'}")
    print("\nTest completed successfully!")
