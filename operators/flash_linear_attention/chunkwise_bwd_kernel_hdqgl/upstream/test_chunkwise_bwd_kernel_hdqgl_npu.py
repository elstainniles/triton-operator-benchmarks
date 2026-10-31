import math
import time
import torch
import torch_npu
import triton
import triton.language as tl
from einops import reduce
from typing import Optional


def ceil_div(x: int, y: int) -> int:
    return math.ceil(x / y)


def ceil_log(x: int, b: int) -> int:
    return math.ceil(math.log(x, b))


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [4]
        for num_stages in [2, 3, 4]
    ],
    key=["H", "K", "V"],
    restore_value=["dq", "dg"],
)
@triton.jit(do_not_specialize=["T"])
def chunkwise_bwd_kernel_hdqgl(
    do,
    q,
    k,
    v,
    g,
    l,
    h_l,
    dq,
    dg,
    dl,
    ell,
    T,
    cu_seqlens,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    L: tl.constexpr,
    BT: tl.constexpr,
    NT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_nh = tl.program_id(0)
    i_n, i_h = i_nh // H, i_nh % H

    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T

    b_h = tl.zeros([V, K], dtype=tl.float32)

    num_intra_levels = (tl.log2(float(BT))).to(tl.int32) + 1

    for i_t in range(tl.cdiv(T, BT)):
        p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        if i_t & (1 << ell):
            p_do = tl.make_block_ptr(
                do + (bos * H + i_h) * V,
                (T, V),
                (H * V, 1),
                (i_t * BT, 0),
                (BT, V),
                (1, 0),
            )
            p_q = tl.make_block_ptr(
                q + bos * K, (T, K), (K, 1), (i_t * BT, 0), (BT, K), (1, 0),
            )
            p_l = tl.make_block_ptr(
                l + (bos * H + i_h) * L + num_intra_levels + ell,
                (T,),
                (H * L,),
                (i_t * BT,),
                (BT,),
                (0,),
            )
            p_dq = tl.make_block_ptr(
                dq + (bos * H + i_h) * K,
                (T, K),
                (H * K, 1),
                (i_t * BT, 0),
                (BT, K),
                (1, 0),
            )
            p_dg = tl.make_block_ptr(
                dg + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,),
            )
            p_dl = tl.make_block_ptr(
                dl + (bos * H + i_h) * L + num_intra_levels + ell,
                (T,),
                (H * L,),
                (i_t * BT,),
                (BT,),
                (0,),
            )
            p_h = tl.make_block_ptr(
                h_l + ((i_n * NT + i_t) * H + i_h) * K * V,
                (V, K),
                (1, V),
                (0, 0),
                (V, K),
                (0, 1),
            )

            b_do = tl.load(p_do, boundary_check=(0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_g = tl.load(p_g, boundary_check=(0,))
            b_l = tl.load(p_l, boundary_check=(0,))

            b_dlq = tl.exp(b_g)[:, None] * tl.dot(b_do, b_h.to(b_do.dtype))

            b_dl = tl.sum(b_dlq * b_q, axis=1)
            b_dg = b_l * b_dl

            tl.store(p_h, b_h, boundary_check=(0, 1))
            b_dq_old = tl.load(p_dq, boundary_check=(0, 1))
            tl.store(
                p_dq,
                (b_l[:, None] * b_dlq).to(p_dq.dtype.element_ty) + b_dq_old,
                boundary_check=(0, 1),
            )
            tl.store(p_dl, b_dl.to(p_dl.dtype.element_ty), boundary_check=(0,))
            b_dg_old = tl.load(p_dg, boundary_check=(0,))
            tl.store(
                p_dg, b_dg.to(p_dg.dtype.element_ty) + b_dg_old, boundary_check=(0,),
            )
            if ((i_t + 1) & (1 << ell)) == 0:
                b_h = tl.zeros([V, K], dtype=tl.float32)

        last_idx = min((i_t + 1) * BT, T) - 1
        b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
        b_h *= tl.exp(b_g_last)
        if (i_t & (1 << ell)) == 0:
            p_k = tl.make_block_ptr(
                k + bos * K, (T, K), (K, 1), (i_t * BT, 0), (BT, K), (1, 0),
            )
            p_v = tl.make_block_ptr(
                v + (bos * H + i_h) * V,
                (V, T),
                (1, H * V),
                (0, i_t * BT),
                (V, BT),
                (0, 1),
            )
            b_g = tl.load(p_g, boundary_check=(0,))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_v = tl.load(p_v, boundary_check=(0, 1))
            b_k = (b_k * tl.exp(b_g_last - b_g)[:, None]).to(b_k.dtype)
            b_h += tl.dot(b_v, b_k)


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


def test_chunkwise_bwd_kernel_hdqgl():
    ptfile_path = "/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunkwise_bwd_kernel_hdqgl/test_chunkwise_bwd_kernel_hdqgl_cuda.pt"
    data = torch.load(ptfile_path, map_location=torch.device('cpu'), weights_only=False)

    device = torch.device('npu')
    input_data = convert_tensor_with_device_type(data["input_data"], device_type='npu')

    do = input_data['do']
    q = input_data['q']
    k = input_data['k']
    v = input_data['v']
    g = input_data['g']
    l = input_data['l']
    h_l = input_data['h_l']
    dq = input_data['dq']
    dg = input_data['dg']
    dl = input_data['dl']
    cu_seqlens = input_data['cu_seqlens']
    ell = input_data['ell']
    B = input_data['B']
    T = input_data['T']
    K = input_data['K']
    H = input_data['H']
    V = input_data['V']
    L = input_data['L']
    BT = input_data['BT']
    NT = input_data['NT']

    grid = (B * H,)

    chunkwise_bwd_kernel_hdqgl[grid](
        do=do,
        q=q,
        k=k,
        v=v,
        g=g,
        l=l,
        h_l=h_l,
        dq=dq,
        dg=dg,
        dl=dl,
        cu_seqlens=cu_seqlens,
        ell=ell,
        T=T,
        H=H,
        K=K,
        V=V,
        L=L,
        BT=BT,
        NT=NT,
    )

    npu_out = {}
    npu_out['dq'] = dq
    npu_out['dg'] = dg
    npu_out['dl'] = dl

    try:
        compare_data_precision(data["gpu_output"], npu_out, device_type='cpu')
        print("Test chunkwise_bwd_kernel_hdqgl passed!")
    except (AssertionError, ValueError, RuntimeError) as e:
        print(f"Precision comparison failed: {e}")
        print("Falling back to fp64 third-party precision comparison...")
        fp64_path = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunkwise_bwd_kernel_hdqgl/test_chunkwise_bwd_kernel_hdqgl_cpu_ref.pt'
        fp64_data = torch.load(fp64_path, map_location=torch.device('cpu'), weights_only=False)
        fp64_precision_compare(npu_out, data["gpu_output"], fp64_data)


def prof_chunkwise_bwd_kernel_hdqgl():
    ptfile_path = "/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunkwise_bwd_kernel_hdqgl/test_chunkwise_bwd_kernel_hdqgl_cuda.pt"
    shape_start = time.time()
    data = torch.load(ptfile_path, map_location=torch.device('cpu'), weights_only=False)

    device = torch.device('npu')
    input_data = convert_tensor_with_device_type(data["input_data"], device_type='npu')

    do = input_data['do']
    q = input_data['q']
    k = input_data['k']
    v = input_data['v']
    g = input_data['g']
    l = input_data['l']
    h_l = input_data['h_l']
    dq = input_data['dq']
    dg = input_data['dg']
    dl = input_data['dl']
    cu_seqlens = input_data['cu_seqlens']
    ell = input_data['ell']
    B = input_data['B']
    T = input_data['T']
    K = input_data['K']
    H = input_data['H']
    V = input_data['V']
    L = input_data['L']
    BT = input_data['BT']
    NT = input_data['NT']

    grid = (B * H,)

    for _ in range(5):
        chunkwise_bwd_kernel_hdqgl[grid](
            do=do,
            q=q,
            k=k,
            v=v,
            g=g,
            l=l,
            h_l=h_l,
            dq=dq,
            dg=dg,
            dl=dl,
            cu_seqlens=cu_seqlens,
            ell=ell,
            T=T,
            H=H,
            K=K,
            V=V,
            L=L,
            BT=BT,
            NT=NT,
        )
        torch_npu.npu.synchronize()

    prof_dir = f"prof_chunkwise_bwd_kernel_hdqgl/shape_{B}_{H}_{T}_{K}"

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
            chunkwise_bwd_kernel_hdqgl[grid](
                do=do,
                q=q,
                k=k,
                v=v,
                g=g,
                l=l,
                h_l=h_l,
                dq=dq,
                dg=dg,
                dl=dl,
                cu_seqlens=cu_seqlens,
                ell=ell,
                T=T,
                H=H,
                K=K,
                V=V,
                L=L,
                BT=BT,
                NT=NT,
            )
            torch_npu.npu.synchronize()
            prof.step()
        prof.stop()

    shape_elapsed = time.time() - shape_start
    print(f"Profile saved to: {prof_dir} (took {shape_elapsed:.1f}s)")


if __name__ == "__main__":
    test_chunkwise_bwd_kernel_hdqgl()
    prof_chunkwise_bwd_kernel_hdqgl()