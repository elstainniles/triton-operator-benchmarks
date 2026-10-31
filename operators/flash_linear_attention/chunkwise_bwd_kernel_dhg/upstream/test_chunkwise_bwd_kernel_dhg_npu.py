import math
import torch
import torch_npu
import triton
import triton.language as tl
from einops import reduce
from typing import Optional
import os
os.environ["TRITON_F32_DEFAULT"] = "ieee"
def ceil_div(x: int, y: int) -> int:
    return math.ceil(x / y)


def ceil_log(x: int, b: int) -> int:
    return math.ceil(math.log(x, b))


BLOCK_K = 64


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[
        triton.Config({"BK": BK}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64, 128]
        for num_warps in [4]
        for num_stages in [2, 3, 4]
    ],
    key=["H", "K", "V"],
    restore_value=["dh", "dg_last"],
)
@triton.jit(do_not_specialize=["T"])
def chunkwise_bwd_kernel_dhg(
    do,
    q,
    g,
    l,
    h_l,
    dh,
    dg_last,
    ell,
    T,
    cu_seqlens,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    L: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    NT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_k = tl.program_id(0)
    i_nh = tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H

    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T

    b_dh = tl.zeros([BK, V], dtype=tl.float32)

    num_intra_levels = (tl.log2(float(BT))).to(tl.int32) + 1

    for i_t in range(tl.cdiv(T, BT) - 1, -1, -1):
        p_dh = tl.make_block_ptr(
            dh + ((i_n * NT + i_t) * H + i_h) * K * V,
            (K, V),
            (V, 1),
            (i_k * BK, 0),
            (BK, V),
            (1, 0),
        )
        b_dh_old = tl.load(p_dh, boundary_check=(0, 1))

        if (i_t & (1 << ell)) == 0:
            tl.store(
                p_dh, b_dh.to(p_dh.dtype.element_ty) + b_dh_old, boundary_check=(0, 1),
            )
            if i_t > 0 and ((i_t - 1) & (1 << ell)) > 0:
                b_dh = tl.zeros([BK, V], dtype=tl.float32)
        if i_t & (1 << ell):
            p_h = tl.make_block_ptr(
                h_l + ((i_n * NT + i_t) * H + i_h) * K * V,
                (K, V),
                (V, 1),
                (i_k * BK, 0),
                (BK, V),
                (1, 0),
            )

            b_h = tl.load(p_h, boundary_check=(0, 1))
            p_dg_last = dg_last + i_n * NT * H + i_t * H + i_h
            tl.atomic_add(p_dg_last, tl.sum(b_h * (b_dh + b_dh_old)))
        last_idx = min((i_t + 1) * BT, T) - 1
        b_g_last = tl.exp(tl.load(g + bos * H + last_idx * H + i_h))
        b_dh *= b_g_last
        if i_t & (1 << ell):
            p_g = tl.make_block_ptr(
                g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,),
            )
            p_q = tl.make_block_ptr(
                q + bos * K, (K, T), (1, K), (i_k * BK, i_t * BT), (BK, BT), (0, 1),
            )
            p_do = tl.make_block_ptr(
                do + (bos * H + i_h) * V,
                (T, V),
                (H * V, 1),
                (i_t * BT, 0),
                (BT, V),
                (1, 0),
            )
            p_l = tl.make_block_ptr(
                l + (bos * H + i_h) * L + num_intra_levels + ell,
                (T,),
                (H * L,),
                (i_t * BT,),
                (BT,),
                (0,),
            )
            b_l = tl.load(p_l, boundary_check=(0,))
            b_g = tl.load(p_g, boundary_check=(0,))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_q = (b_q * (tl.exp(b_g) * b_l)[None, :]).to(b_q.dtype)
            b_do = tl.load(p_do, boundary_check=(0, 1))

            b_s = tl.dot(b_q, b_do).to(b_q.dtype)
            b_dh += b_s


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

def test_chunkwise_bwd_kernel_dhg():
    load_dir = "/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunkwise_bwd_kernel_dhg/"
    ptfile_path = os.path.join(load_dir, "test_chunkwise_bwd_kernel_dhg_cuda.pt")
    data = torch.load(ptfile_path, map_location=torch.device('cpu'), weights_only=False)

    device = torch.device('npu')
    input_data = convert_tensor_with_device_type(data["input_data"], device_type='npu')

    do = input_data['do']
    q = input_data['q']
    g = input_data['g']
    l = input_data['l']
    h_l = input_data['h_l']
    dh = input_data['dh']
    dg_last = input_data['dg_last']
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

    def grid_f(meta):
        return (triton.cdiv(K, meta["BK"]), B * H)

    chunkwise_bwd_kernel_dhg[grid_f](
        do=do,
        q=q,
        g=g,
        l=l,
        h_l=h_l,
        dh=dh,
        dg_last=dg_last,
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
    npu_out['dh'] = dh
    npu_out['dg_last'] = dg_last

    # 三方对比
    fp64_precision_compare(npu_out, data["gpu_output"], data["gpu_output_fp64"])
    print("Test chunkwise_bwd_kernel_dhg passed!")


if __name__ == "__main__":
    test_chunkwise_bwd_kernel_dhg()

