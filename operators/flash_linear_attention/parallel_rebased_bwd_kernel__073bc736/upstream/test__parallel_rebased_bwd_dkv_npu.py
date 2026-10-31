import time
import torch
import torch_npu
import triton
import triton.language as tl
from typing import Optional


@triton.jit(do_not_specialize=['T'])
def _parallel_rebased_bwd_dkv(
    i_bh,
    i_c,
    i_k,
    i_v,
    i_h,
    q,
    k,
    v,
    do,
    dz,
    dk,
    dv,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BTL: tl.constexpr,
    BTS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    p_k = tl.make_block_ptr(k + i_bh * T*K, (T, K), (K, 1), (i_c*BTL, i_k*BK), (BTL, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * T*V, (T, V), (V, 1), (i_c*BTL, i_v*BV), (BTL, BV), (1, 0))
    b_k, b_v = tl.load(p_k, boundary_check=(0, 1)), tl.load(p_v, boundary_check=(0, 1))
    b_dk, b_dv = tl.zeros([BTL, BK], dtype=tl.float32), tl.zeros(
        [BTL, BV], dtype=tl.float32)

    for i in range((tl.cdiv(T, BTS) * BTS)-BTS, (i_c + 1) * BTL - BTS, -BTS):
        p_q = tl.make_block_ptr(q + i_bh * T*K, (K, T), (1, K), (i_k*BK, i), (BK, BTS), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * T*V, (V, T), (1, V), (i_v*BV, i), (BV, BTS), (0, 1))
        p_dz = dz + i_bh * T + i + tl.arange(0, BTS)
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1)).to(b_q.dtype)
        b_dz = tl.load(p_dz, mask=(i + tl.arange(0, BTS)) < T)
        b_s = tl.dot(b_k.to(b_q.dtype), b_q, allow_tf32=False) * scale
        b_s2 = b_s * b_s
        b_dv += tl.dot(b_s2.to(b_q.dtype), tl.trans(b_do), allow_tf32=False)
        b_ds = tl.dot(b_v, b_do, allow_tf32=False) * scale
        if i_v == 0:
            b_ds += b_dz[None, :] * scale
        else:
            b_ds = b_ds
        b_dk += tl.dot((2 * b_ds * b_s).to(b_q.dtype), tl.trans(b_q), allow_tf32=False)

    tl.debug_barrier()
    o_q, o_k = tl.arange(0, BTS), tl.arange(0, BTL)
    for i in range(i_c*BTL, (i_c+1)*BTL, BTS):
        p_q = tl.make_block_ptr(q + i_bh * T*K, (K, T), (1, K), (i_k*BK, i), (BK, BTS), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * T*V, (V, T), (1, V), (i_v*BV, i), (BV, BTS), (0, 1))
        p_dz = dz + i_bh * T + i + tl.arange(0, BTS)
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1)).to(b_q.dtype)
        b_dz = tl.load(p_dz, mask=(i + tl.arange(0, BTS)) < T)
        m_s = o_k[:, None] <= o_q[None, :]
        b_s = tl.dot(b_k, b_q, allow_tf32=False) * scale
        b_s2 = b_s * b_s
        b_s = tl.where(m_s, b_s, 0)
        b_s2 = tl.where(m_s, b_s2, 0)

        b_ds = tl.dot(b_v, b_do, allow_tf32=False)
        if i_v == 0:
            b_ds += b_dz[None, :]
        else:
            b_ds = b_ds
        b_ds = tl.where(m_s, b_ds, 0) * scale
        b_dv += tl.dot(b_s2.to(b_q.dtype), tl.trans(b_do), allow_tf32=False)
        b_dk += tl.dot((2 * b_ds * b_s).to(b_q.dtype), tl.trans(b_q), allow_tf32=False)
        o_q += BTS

    p_dk = tl.make_block_ptr(dk + (i_bh + B * H * i_v) * T*K, (T, K), (K, 1), (i_c*BTL, i_k*BK), (BTL, BK), (1, 0))
    p_dv = tl.make_block_ptr(dv + (i_bh + B * H * i_k) * T*V, (T, V), (V, 1), (i_c*BTL, i_v*BV), (BTL, BV), (1, 0))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
    return


@triton.jit(do_not_specialize=['T'])
def parallel_rebased_bwd_kernel(
    q,
    k,
    v,
    do,
    dz,
    dq,
    dk,
    dv,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BTL: tl.constexpr,
    BTS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_kv, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    NV = tl.cdiv(V, BV)
    i_k = i_kv // (NV)
    i_v = i_kv % (NV)
    i_h = i_bh % H
    tl.debug_barrier()
    _parallel_rebased_bwd_dkv(
        i_bh,
        i_c,
        i_k,
        i_v,
        i_h,
        q,
        k,
        v,
        do,
        dz,
        dk,
        dv,
        scale,
        B=B,
        H=H,
        T=T,
        K=K,
        V=V,
        BTL=BTL,
        BTS=BTS,
        BK=BK,
        BV=BV,
    )


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
        torch.testing.assert_close(y_ref, y_cal, rtol=1e-5, atol=1e-5, equal_nan=True)
    elif dtype == 'int32' or dtype == 'int64' or dtype == 'int16' or dtype == 'int8':
        assert torch.equal(y_cal, y_ref)
    elif dtype == 'bool':
        assert torch.equal(y_cal, y_ref)
    else:
        raise ValueError('Invalid parameter "dtype" is found : {}'.format(dtype))


def compare_data_precision(dict_ref: dict, dict_cal: dict, device_type: str):
    keys_ref, keys_cal = set(dict_ref.keys()), set(dict_cal.keys())
    if not keys_ref.issubset(keys_cal):
        raise ValueError("The keys of dict_ref is not subset of dict_cal")

    for key in dict_ref.keys():
        val_a, val_b = dict_ref[key], dict_cal[key]
        if val_a is None and val_b is None:
            continue
        if type(val_a) != type(val_b):
            raise ValueError("The data type of two dicts are different")

        if isinstance(val_a, torch.Tensor):
            validate_cmp(dtype=str(val_a.dtype).split('.')[-1], y_ref=val_a, y_cal=val_b, device_type=device_type)
        else:
            raise ValueError("Non-tensor type is not currently supported")


def convert_tensor_with_device_type(indata: dict, device_type: str):
    target_device = torch.device(device_type)
    outdata = {}

    for key, value in indata.items():
        if isinstance(value, torch.Tensor):
            if value.device.type != target_device.type:
                outdata[key] = value.to(target_device)
            else:
                outdata[key] = value
        else:
            outdata[key] = value

    return outdata


def cal_precision(input: torch.Tensor, golden: torch.Tensor):
    relative_max = torch.max(torch.abs(input - golden) / (torch.abs(golden) + float(1e-7)))
    relative_mean = torch.mean(torch.abs(input - golden) / (torch.abs(golden) + float(1e-7)))
    rmse = torch.sqrt(torch.mean((input - golden) ** 2))
    mask = (torch.abs(golden) < float(2e-14)) & (torch.abs(input - golden) > float(2e-30))
    error = torch.abs(input - golden)
    total_error = torch.sum(error[mask])
    return {
        'max_relative_error': relative_max.item(),
        'mean_relative_error': relative_mean.item(),
        'rmse': rmse.item(),
        'small_val_error': total_error.item(),
    }


def fp64_precision_compare(npu_out, gpu_out, fp64_data):
    print("\n" + "=" * 60)
    print("Third-party fp64 precision comparison")
    print("=" * 60)
    for key in fp64_data.keys():
        if not isinstance(fp64_data[key], torch.Tensor):
            continue
        if fp64_data[key].is_floating_point():
            fp64_val = fp64_data[key].to(torch.float64).cpu()
            npu_val = npu_out[key].to(torch.float64).cpu()
            gpu_val = gpu_out[key].to(torch.float64).cpu()

            npu_metrics = cal_precision(npu_val, fp64_val)
            gpu_metrics = cal_precision(gpu_val, fp64_val)

            print(f"\n--- Key: {key} ---")
            print(f"{'Metric':<25} {'NPU vs FP64':<20} {'GPU vs FP64':<20} {'NPU/GPU Ratio':<15}")
            print("-" * 80)
            for metric_name in npu_metrics:
                npu_v = npu_metrics[metric_name]
                gpu_v = gpu_metrics[metric_name]
                if gpu_v != 0:
                    ratio = npu_v / gpu_v
                    ratio_str = f"{ratio:.6f}"
                else:
                    ratio_str = "N/A (gpu=0)"
                print(f"{metric_name:<25} {npu_v:<20.10f} {gpu_v:<20.10f} {ratio_str:<15}")
                if metric_name == "max_relative_error":
                     assert ratio < 2.0, f"Ratio {ratio} exceeds threshold 2.0 for {metric_name}"
                elif metric_name != "small_val_error":
                     assert ratio < 1.2, f"Ratio {ratio} exceeds threshold 1.2 for {metric_name}"


def test_parallel_rebased_bwd_dkv():
    ptfile_path = "/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/_parallel_rebased_bwd_dkv/test_parallel_rebased_bwd_dkv_cuda.pt"
    data = torch.load(ptfile_path, map_location=torch.device('cpu'), weights_only=False)

    device = torch.device('npu')
    input_data = convert_tensor_with_device_type(data["input_data"], device_type='npu')

    B = input_data['B']
    H = input_data['H']
    T = input_data['T']
    K = input_data['K']
    V = input_data['V']
    BTL = input_data['BTL']
    BTS = input_data['BTS']
    BK = input_data['BK']
    BV = input_data['BV']
    scale = input_data['scale']

    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)

    dq = torch.empty(NV, B, H, T, K, dtype=torch.float32, device=device)
    dk = torch.empty(NV, B, H, T, K, dtype=torch.float32, device=device)
    dv = torch.empty(NK, B, H, T, V, dtype=torch.float32, device=device)

    input_data['dq'] = dq
    input_data['dk'] = dk
    input_data['dv'] = dv

    grid = (NK * NV, triton.cdiv(T, BTL), B * H)

    parallel_rebased_bwd_kernel[grid](
        input_data['q'],
        input_data['k'],
        input_data['v'],
        input_data['do'],
        input_data['dz'],
        input_data['dq'],
        input_data['dk'],
        input_data['dv'],
        scale,
        T=T,
        B=B,
        H=H,
        K=K,
        V=V,
        BTL=BTL,
        BTS=BTS,
        BK=BK,
        BV=BV,
        num_warps=4,
        num_stages=2,
    )

    npu_out = {}
    npu_out['dk'] = dk
    npu_out['dv'] = dv

    try:
        compare_data_precision(data["gpu_output"], npu_out, device_type='cpu')
        print("Test _parallel_rebased_bwd_dkv passed!")
    except (AssertionError, ValueError, RuntimeError) as e:
        print(f"Precision comparison failed: {e}")
        print("Falling back to fp64 third-party precision comparison...")
        fp64_path = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/_parallel_rebased_bwd_dkv/test_parallel_rebased_bwd_dkv_cuda_fp64.pt'
        fp64_data = torch.load(fp64_path, map_location=torch.device('cpu'), weights_only=False)
        fp64_precision_compare(npu_out, data["gpu_output"], fp64_data["gpu_output"])

def prof_parallel_rebased_bwd_dkv():
    ptfile_path = "/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/_parallel_rebased_bwd_dkv/test_parallel_rebased_bwd_dkv_cuda.pt"
    shape_start = time.time()
    data = torch.load(ptfile_path, map_location=torch.device('cpu'), weights_only=False)

    device = torch.device('npu')
    input_data = convert_tensor_with_device_type(data["input_data"], device_type='npu')

    B = input_data['B']
    H = input_data['H']
    T = input_data['T']
    K = input_data['K']
    V = input_data['V']
    BTL = input_data['BTL']
    BTS = input_data['BTS']
    BK = input_data['BK']
    BV = input_data['BV']
    scale = input_data['scale']

    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)

    dq = torch.empty(NV, B, H, T, K, dtype=torch.float32, device=device)
    dk = torch.empty(NV, B, H, T, K, dtype=torch.float32, device=device)
    dv = torch.empty(NK, B, H, T, V, dtype=torch.float32, device=device)

    input_data['dq'] = dq
    input_data['dk'] = dk
    input_data['dv'] = dv

    grid = (NK * NV, triton.cdiv(T, BTL), B * H)

    # --- Warmup ---
    for _ in range(5):
        parallel_rebased_bwd_kernel[grid](
            input_data['q'],
            input_data['k'],
            input_data['v'],
            input_data['do'],
            input_data['dz'],
            input_data['dq'],
            input_data['dk'],
            input_data['dv'],
            scale,
            T=T,
            B=B,
            H=H,
            K=K,
            V=V,
            BTL=BTL,
            BTS=BTS,
            BK=BK,
            BV=BV,
            num_warps=4,
            num_stages=2
        )
        torch_npu.npu.synchronize()

    # --- NPU Profiling ---
    prof_dir = f"prof_parallel_rebased_bwd_dkv/shape_{B}_{H}_{T}_{K}"

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
            parallel_rebased_bwd_kernel[grid](
                input_data['q'],
                input_data['k'],
                input_data['v'],
                input_data['do'],
                input_data['dz'],
                input_data['dq'],
                input_data['dk'],
                input_data['dv'],
                scale,
                T=T,
                B=B,
                H=H,
                K=K,
                V=V,
                BTL=BTL,
                BTS=BTS,
                BK=BK,
                BV=BV,
                num_warps=4,
                num_stages=2
            )
            torch_npu.npu.synchronize()
            prof.step()
        prof.stop()

    shape_elapsed = time.time() - shape_start
    print(f"Profile saved to: {prof_dir} (took {shape_elapsed:.1f}s)")


if __name__ == "__main__":
    test_parallel_rebased_bwd_dkv()
    prof_parallel_rebased_bwd_dkv()
