import time
import torch
import triton
import triton.language as tl
import torch_npu
from typing import Optional


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
        torch.testing.assert_close(y_ref.to(torch.float32), y_cal.to(torch.float32), rtol=1e-02, atol=1e-03,
                                   equal_nan=True)
    elif dtype == 'float32':
        torch.testing.assert_close(y_ref, y_cal, rtol=1e-05, atol=1e-05, equal_nan=True)
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


@triton.jit
def fused_chunk_based_bwd_kernel(
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
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] >= o_i[None, :]

    b_h_1o = tl.zeros([BV, BK], dtype=tl.float32)
    b_h_2o = tl.zeros([BV, BK*BK], dtype=tl.float32)

    k_1o = tl.zeros([1, BK], dtype=tl.float32)
    k_2o = tl.zeros([1, BK * BK], dtype=tl.float32)

    for i in range(0, tl.cdiv(T, BT)):
        p_q = tl.make_block_ptr(q + i_bh * T*K, (T, K), (K, 1), (i * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k + i_bh * T*K, (T, K), (K, 1), (i * BT, i_k * BK), (BT, BK), (1, 0))
        p_v = tl.make_block_ptr(v + i_bh * T*V, (V, T), (1, V), (i_v * BV, i * BT), (BV, BT), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * T*V, (T, V), (V, 1), (i * BT, i_v * BV), (BT, BV), (1, 0))
        p_dq = tl.make_block_ptr(dq + (i_bh + i_v*B*H) * T*K, (T, K), (K, 1), (i*BT, i_k*BK), (BT, BK), (1, 0))
        p_dz = dz + (i_bh) * T + tl.arange(0, BT) + i * BT
        b_dq = tl.zeros([BT, BK], dtype=tl.float32)

        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1)).to(b_q.dtype)
        b_dz = tl.load(p_dz, mask=(tl.arange(0, BT) + i * BT) < T)
        b_v = tl.load(p_v, boundary_check=(0, 1))

        b_dq += tl.dot(b_do, (b_h_1o).to(b_do.dtype), allow_tf32=False)
        if i_v == 0:
            b_dq += b_dz[:, None] * k_1o
        b_dq_2o = tl.dot(b_do, (b_h_2o).to(b_do.dtype), allow_tf32=False) * 0.5
        if i_v == 0:
            b_dq_2o += (b_dz[:, None] * k_2o) * 0.5
        b_dq_2o = tl.reshape(b_dq_2o, [BT, BK, BK])
        b_dq += tl.sum(b_dq_2o * b_q[:, :, None], axis=1)
        b_dq += tl.sum(b_dq_2o * b_q[:, None, :], axis=2)
        b_dq *= scale

        b_ds = tl.dot(b_do, b_v, allow_tf32=False)
        if i_v == 0:
            b_ds += b_dz[:, None]
        b_ds = tl.where(m_s, b_ds, 0) * scale
        b_s = tl.dot(b_q, tl.trans(b_k), allow_tf32=False)
        b_s = tl.where(m_s, b_s, 0)
        b_dq += tl.dot((b_ds * (1 + b_s)).to(b_q.dtype), b_k, allow_tf32=False)

        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))

        b_k_2o = b_k[:, :, None] * b_k[:, None, :]
        b_k_2o = tl.reshape(b_k_2o, [BT, BK * BK]).to(b_k.dtype)
        b_h_2o = b_h_2o + tl.dot(b_v, b_k_2o.to(b_v.dtype), allow_tf32=False)
        b_h_1o = b_h_1o + tl.dot(b_v, b_k, allow_tf32=False)

        if i_v == 0:
            k_1o += tl.sum(b_k, axis=0)[None, :]
            k_2o += tl.sum(b_k_2o, axis=0)[None, :]

    tl.debug_barrier()
    b_h_1o = None
    b_h_2o = None

    b_dh_1o = tl.zeros([BK, BV], dtype=tl.float32)
    b_dh_2o = tl.zeros([BK*BK, BV], dtype=tl.float32)
    b_dh_0o = tl.zeros([BV], dtype=tl.float32)
    m_s = tl.arange(0, BT)[:, None] <= tl.arange(0, BT)[None, :]

    dq_1o = tl.zeros([1, BK], dtype=tl.float32)
    dq_2o = tl.zeros([BK * BK, 1], dtype=tl.float32)

    for i in range(tl.cdiv(T, BT) * BT - BT, -BT, -BT):
        p_q = tl.make_block_ptr(q + i_bh * T*K, (K, T), (1, K), (i_k * BK, i), (BK, BT), (0, 1))
        p_k = tl.make_block_ptr(k + i_bh * T*K, (T, K), (K, 1), (i, i_k * BK), (BT, BK), (1, 0))
        p_v = tl.make_block_ptr(v + i_bh * T*V, (T, V), (V, 1), (i, i_v * BV), (BT, BV), (1, 0))
        p_do = tl.make_block_ptr(do + i_bh * T*V, (T, V), (V, 1), (i, i_v * BV), (BT, BV), (1, 0))
        p_dk = tl.make_block_ptr(dk + (i_bh+i_v*B*H) * T*K, (T, K), (K, 1), (i, i_k*BK), (BT, BK), (1, 0))
        p_dv = tl.make_block_ptr(dv + (i_bh+i_k*B*H) * T*V, (T, V), (V, 1), (i, i_v*BV), (BT, BV), (1, 0))
        p_dz = dz + (i_bh) * T + tl.arange(0, BT) + i

        b_dk = tl.zeros([BT, BK], dtype=tl.float32)
        b_dv = tl.zeros([BT, BV], dtype=tl.float32)

        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1)).to(b_q.dtype)
        b_dz = tl.load(p_dz, mask=(tl.arange(0, BT)+i) < T)
        b_q = (b_q * scale).to(b_k.dtype)

        b_ds = tl.dot(b_v, tl.trans(b_do), allow_tf32=False)
        if i_v == 0:
            b_ds += b_dz[None, :]
        b_ds = tl.where(m_s, b_ds, 0)
        b_s = tl.dot(b_k, b_q, allow_tf32=False)
        b_s2 = 1 + b_s + 0.5 * b_s * b_s
        b_s = tl.where(m_s, b_s, 0)
        b_s2 = tl.where(m_s, b_s2, 0)
        b_ds *= (1+b_s)

        b_dk += tl.dot(b_ds.to(b_k.dtype), tl.trans(b_q), allow_tf32=False)
        b_dv += tl.dot(b_s2.to(b_do.dtype), b_do, allow_tf32=False)

        b_k_2o = b_k[:, :, None] * b_k[:, None, :]
        b_k_2o = tl.reshape(b_k_2o, [BT, BK * BK]).to(b_k.dtype)

        b_dv += tl.dot(b_k, b_dh_1o.to(b_k.dtype), allow_tf32=False)
        b_dv += tl.dot(b_k_2o, b_dh_2o.to(b_k.dtype), allow_tf32=False)
        b_dv += b_dh_0o

        b_dk += tl.dot(b_v, tl.trans(b_dh_1o).to(b_k.dtype), allow_tf32=False)

        if i_v == 0:
            b_dk += dq_1o

        b_dk_2o = tl.dot(b_dh_2o.to(b_k.dtype), tl.trans(b_v), allow_tf32=False)
        if i_v == 0:
            b_dk_2o += dq_2o
        b_dk_2o = tl.reshape(b_dk_2o, [BK, BK, BT])
        b_k_fp32 = tl.trans(b_k.to(tl.float32))
        b_dk2 = tl.sum(b_dk_2o * b_k_fp32[:, None, :], axis=0)
        b_dk2 += tl.sum(b_dk_2o * b_k_fp32[None, :, :], axis=1)
        b_dk += tl.trans(b_dk2)

        b_dh_0o += tl.sum(b_do, axis=0)
        b_dh_1o = b_dh_1o + tl.dot(b_q, b_do, allow_tf32=False)
        b_q_2o = b_q[None, :, :] * b_q[:, None, :]
        b_q_2o = tl.reshape(b_q_2o, [BK * BK, BT]).to(b_k.dtype)
        b_dh_2o = b_dh_2o + tl.dot(b_q_2o, b_do, allow_tf32=False) * 0.5

        if i_v == 0:
            dq_1o += (tl.sum(b_dz[None, :] * b_q, axis=1))[None, :]
            dq_2o += (tl.sum(b_dz[None, :] * b_q_2o, axis=1) * 0.5)[:, None]

        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


class Model(torch.nn.Module):
    def __init__(self, B, H, T, K, V, scale):
        super(Model, self).__init__()
        self.B = B
        self.H = H
        self.T = T
        self.K = K
        self.V = V
        self.scale = scale

    def forward(self, q, k, v, do, dz):
        B, H, T, K, V = self.B, self.H, self.T, self.K, self.V
        scale = self.scale

        BT = 16
        BK, BV = min(K, 16), min(V, 32)
        BK, BV = max(BK, 16), max(BV, 16)
        NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
        num_stages = 1
        num_warps = 4

        dq = q.new_empty(NV, B, H, T, K)
        dk = q.new_empty(NV, B, H, T, K)
        dv = q.new_empty(NK, B, H, T, V)
        grid = (NV, NK, B * H)

        fused_chunk_based_bwd_kernel[grid](
            q, k, v, do, dz, dq, dk, dv,
            scale,
            T=T, B=B, H=H, K=K, V=V, BT=BT, BK=BK, BV=BV,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        dq = dq.sum(0)
        dk = dk.sum(0)
        dv = dv.sum(0)
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)


def test_fused_chunk_based_backward():
    ptfile_path = "/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/fused_chunk_based_bwd_kernel/test_fused_chunk_based_bwd_kernel_cuda.pt"
    data = torch.load(ptfile_path, map_location=torch.device('cpu'), weights_only=False)

    input_data = convert_tensor_with_device_type(data["input_data"], device_type='npu')

    q = input_data['q']
    k = input_data['k']
    v = input_data['v']
    do = input_data['do']
    dz = input_data['dz']
    B = input_data['B']
    H = input_data['H']
    T = input_data['T']
    K = input_data['K']
    V = input_data['V']
    scale = input_data['scale']

    print(f"Testing FusedChunkBasedFunction_backward on NPU")
    print(f"Input shapes: q={q.shape}, k={k.shape}, v={v.shape}")
    print(f"             do={do.shape}, dz={dz.shape}")

    model = Model(B, H, T, K, V, scale)

    with torch.no_grad():
        dq, dk, dv = model(q, k, v, do, dz)

    print(f"Output shapes: dq={dq.shape}, dk={dk.shape}, dv={dv.shape}")

    npu_out = {
        'dq': dq.cpu(),
        'dk': dk.cpu(),
        'dv': dv.cpu(),
    }

    try:
        compare_data_precision(data["gpu_output"], npu_out, device_type='cpu')
        print("Test fused_chunk_based_bwd_kernel passed!")
    except (AssertionError, ValueError, RuntimeError) as e:
        print(f"Precision comparison failed: {e}")
        print("Falling back to fp64 third-party precision comparison...")
        fp64_path = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/fused_chunk_based_bwd_kernel/test_fused_chunk_based_bwd_kernel_cuda_fp64.pt'
        fp64_data = torch.load(fp64_path, map_location=torch.device('cpu'), weights_only=False)
        fp64_precision_compare(npu_out, data["gpu_output"], fp64_data["gpu_output"])


def prof_fused_chunk_based_bwd_kernel():
    ptfile_path = "/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/fused_chunk_based_bwd_kernel/test_fused_chunk_based_bwd_kernel_cuda.pt"
    shape_start = time.time()
    data = torch.load(ptfile_path, map_location=torch.device('cpu'), weights_only=False)

    input_data = convert_tensor_with_device_type(data["input_data"], device_type='npu')

    q = input_data['q']
    k = input_data['k']
    v = input_data['v']
    do = input_data['do']
    dz = input_data['dz']
    B = input_data['B']
    H = input_data['H']
    T = input_data['T']
    K = input_data['K']
    V = input_data['V']
    scale = input_data['scale']

    model = Model(B, H, T, K, V, scale)

    for _ in range(5):
        with torch.no_grad():
            model(q, k, v, do, dz)
        torch_npu.npu.synchronize()

    prof_dir = f"prof_fused_chunk_based_bwd_kernel/shape_{B}_{H}_{T}_{K}"

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
            with torch.no_grad():
                model(q, k, v, do, dz)
            torch_npu.npu.synchronize()
            prof.step()
        prof.stop()

    shape_elapsed = time.time() - shape_start
    print(f"Profile saved to: {prof_dir} (took {shape_elapsed:.1f}s)")


if __name__ == '__main__':
    test_fused_chunk_based_backward()
    prof_fused_chunk_based_bwd_kernel()