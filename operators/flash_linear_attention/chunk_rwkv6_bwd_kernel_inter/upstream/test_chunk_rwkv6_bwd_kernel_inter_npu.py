# coding=utf-8
import sys, os, pytest, triton, torch, triton.language as tl, time, torch_npu
from typing import Optional
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@triton.jit
def exp(x):
    return tl.exp(x.to(tl.float32))

@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.jit(do_not_specialize=['T'])
def chunk_rwkv6_bwd_kernel_inter(q, k, v, h, gi, ge, u, do, dh, dA, dq, dk, dq2, dk2, dg, du,
    cu_seqlens, chunk_indices, scale, T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, IS_VARLEN: tl.constexpr):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T
    o_k = i_k * BK + tl.arange(0, BK)
    m_k = o_k < K
    p_gk = tl.make_block_ptr(ge + (bos*H+i_h)*K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_gi = tl.make_block_ptr(gi + (bos*H+i_h)*K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_gn = gi + (bos + min(T, i_t * BT + BT)-1) * H*K + i_h * K + o_k
    b_gn = tl.load(p_gn, mask=m_k, other=0)
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_dgk = tl.zeros([BK], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_do = tl.make_block_ptr(do + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + (i_tg * H + i_h) * K*V, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        p_dh = tl.make_block_ptr(dh + (i_tg * H + i_h) * K*V, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_dh = tl.load(p_dh, boundary_check=(0, 1))
        b_dgk += tl.sum(b_h * b_dh, axis=0)
        b_dq += tl.dot(b_do, b_h.to(b_do.dtype))
        b_dk += tl.dot(b_v, b_dh.to(b_v.dtype))
    b_dgk *= exp(b_gn)
    b_dq *= scale
    b_gk = tl.load(p_gk, boundary_check=(0, 1))
    b_gi = tl.load(p_gi, boundary_check=(0, 1))
    b_dq = b_dq * exp(b_gk)
    b_dk = b_dk * exp(b_gn[None, :] - b_gi)
    o_i = tl.arange(0, BT)
    p_q = tl.make_block_ptr(q + (bos*H+i_h)*K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + (bos*H+i_h)*K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dq = tl.make_block_ptr(dq + (bos*H+i_h)*K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk + (bos*H+i_h)*K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dA_dig = dA + ((bos + i_t * BT + o_i) * H + i_h) * BT + o_i
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_dgk += tl.sum(b_dk * b_k, axis=0)
    b_dq += tl.load(p_dq, boundary_check=(0, 1))
    b_dk += tl.load(p_dk, boundary_check=(0, 1))
    b_dg = b_q * b_dq - b_k * b_dk
    g0 = b_q * b_dq - b_k * b_dk
    tail = tl.cumsum(g0, axis=0, reverse=True)
    b_dg = tail + b_dgk[None, :] - b_q * b_dq
    b_dA_dig = tl.load(p_dA_dig, mask=(i_t * BT + o_i) < T, other=0)
    p_u = tl.make_block_ptr(u + i_h * K, (K,), (1,), (i_k * BK,), (BK,), (0,))
    b_u = tl.load(p_u, boundary_check=(0,))
    b_dq += (b_dA_dig[:, None] * b_u[None, :] * b_k)
    b_dk += (b_dA_dig[:, None] * b_u[None, :] * b_q)
    b_du = tl.sum(b_dA_dig[:, None] * b_q * b_k, axis=0)
    p_du = tl.make_block_ptr(du + (i_tg * H + i_h) * K, (K,), (1,), (i_k * BK,), (BK,), (0,))
    tl.store(p_du, b_du, boundary_check=(0,))
    p_dq = tl.make_block_ptr(dq2 + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk2 + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dg = tl.make_block_ptr(dg + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0, 1))

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

def test_chunk_rwkv6_bwd_kernel_inter_npu():
    ptfile_path = "/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_rwkv6_bwd_kernel_inter/test_chunk_rwkv6_bwd_kernel_inter.pt"
    data = torch.load(ptfile_path, map_location=torch.device('cpu'), weights_only=False)
    input_data = convert_tensor_with_device_type(data["input_data"], device_type='npu')
    chunk_rwkv6_bwd_kernel_inter[data["grid"]](
        q=input_data['q'], k=input_data['k'], v=input_data['v'], h=input_data['h'],
        gi=input_data['gi'], ge=input_data['ge'], u=input_data['u'], do=input_data['do'],
        dh=input_data['dh'], dA=input_data['dA'], dq=input_data['dq'], dk=input_data['dk'],
        dq2=input_data['dq2'], dk2=input_data['dk2'], dg=input_data['dg'], du=input_data['du'],
        cu_seqlens=input_data['cu_seqlens'] if input_data['cu_seqlens'].numel() > 0 else None,
        chunk_indices=input_data['chunk_indices'] if input_data['chunk_indices'].numel() > 0 else None,
        scale=input_data['scale'], T=input_data['T'], H=input_data['H'], K=input_data['K'],
        V=input_data['V'], BT=input_data['BT'], BK=input_data['BK'], BV=input_data['BV'],
    )
    npu_output = {
        'dq2': input_data['dq2'].cpu().clone(),
        'dk2': input_data['dk2'].cpu().clone(),
        'dg': input_data['dg'].cpu().clone(),
        'du': input_data['du'].cpu().clone()
    }
    try:
        golden_data = torch.load('/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_rwkv6_bwd_kernel_inter/test_chunk_rwkv6_bwd_kernel_inter_fp64_v2.pt', map_location='cpu', weights_only=False)
        golden_output = golden_data['gpu_output']
    except Exception as e:
        pytest.fail(f"load golden file failed: {str(e)}")
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

    def cal_precision(input, golden):
        relative_max = torch.max(torch.abs(input - golden) / (torch.abs(golden) + float(1e-7)))
        relative_mean = torch.mean(torch.abs(input - golden) / (torch.abs(golden) + float(1e-7)))
        rmse = torch.sqrt(torch.mean((input - golden) ** 2))
        mask = (torch.abs(golden) < float(2e-14)) & (torch.abs(input - golden) > float(2e-30))
        small_val_error = torch.sum(torch.abs(input - golden)[mask])
        return relative_max.item(), relative_mean.item(), rmse.item(), small_val_error.item()

    gpu_fp32_output = None
    try:
        gpu_fp32_data = torch.load('/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_rwkv6_bwd_kernel_inter/test_chunk_rwkv6_bwd_kernel_inter_v2.pt', map_location='cpu', weights_only=False)
        gpu_fp32_output = gpu_fp32_data['gpu_output']
    except Exception as e:
        print(f"Could not load GPU fp32 data: {e}")

    print("\n" + "="*60)
    print("NPU vs GPU fp64 Validation & Comparison")
    print("="*60)
    for key in ['dq2', 'dk2', 'dg', 'du']:
        dtype_str = str(npu_output[key].dtype).replace('torch.', '')
        try:
            validate_cmp(dtype_str, npu_output[key].clone(), golden_output[key].float().clone())
            print(f"\n{key}: PASSED (validate_cmp)")
        except (AssertionError, RuntimeError) as e:
            print(f"\n{key}: FAILED validate_cmp - {e}")
            npu_rel_max, npu_rel_mean, npu_rmse, npu_small = cal_precision(npu_output[key].double(), golden_output[key].double())
            print(f"  NPU vs GPU fp64:")
            print(f"    Max relative error:  {npu_rel_max:.6e}")
            print(f"    Mean relative error: {npu_rel_mean:.6e}")
            print(f"    RMSE:                {npu_rmse:.6e}")
            print(f"    Small val abs error: {npu_small:.6e}")
            if gpu_fp32_output is not None:
                gpu_rel_max, gpu_rel_mean, gpu_rmse, gpu_small = cal_precision(gpu_fp32_output[key].double(), golden_output[key].double())
                print(f"  GPU fp32 vs GPU fp64:")
                print(f"    Max relative error:  {gpu_rel_max:.6e}")
                print(f"    Mean relative error: {gpu_rel_mean:.6e}")
                print(f"    RMSE:                {gpu_rmse:.6e}")
                print(f"    Small val abs error: {gpu_small:.6e}")
                print(f"  NPU/GPU fp32 ratio:")
                print(f"    Max relative error:  {npu_rel_max / (gpu_rel_max + 1e-10):.2f}x")
                print(f"    Mean relative error: {npu_rel_mean / (gpu_rel_mean + 1e-10):.2f}x")
                print(f"    RMSE:                {npu_rmse / (gpu_rmse + 1e-10):.2f}x")
                print(f"    Small val abs error: {npu_small / (gpu_small + 1e-10):.2f}x")
                assert npu_rel_max / (gpu_rel_max + 1e-10) < 2, f"{key} Max relative error NPU/GPU fp32 ratio too high"
                assert npu_rel_mean / (gpu_rel_mean + 1e-10) < 1.2, f"{key} Mean relative error NPU/GPU fp32 ratio too high"
                assert npu_rmse / (gpu_rmse + 1e-10) < 1.2, f"{key} RMSE NPU/GPU fp32 ratio too high"
                print(f"\n{key}: PASSED (third_cmp)")

    print("="*60 + "\n")

def prof_chunk_rwkv6_bwd_kernel_inter():
    ptfile_path = "/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_rwkv6_bwd_kernel_inter/test_chunk_rwkv6_bwd_kernel_inter.pt"
    shape_start = time.time()
    data = torch.load(ptfile_path, map_location=torch.device('cpu'), weights_only=False)
    input_data = convert_tensor_with_device_type(data["input_data"], device_type='npu')

    T = input_data['T']
    H = input_data['H']
    K = input_data['K']
    V = input_data['V']
    BT = input_data['BT']

    grid = data["grid"]

    # --- Warmup ---
    for _ in range(5):
        chunk_rwkv6_bwd_kernel_inter[grid](
            q=input_data['q'], k=input_data['k'], v=input_data['v'], h=input_data['h'],
            gi=input_data['gi'], ge=input_data['ge'], u=input_data['u'], do=input_data['do'],
            dh=input_data['dh'], dA=input_data['dA'], dq=input_data['dq'], dk=input_data['dk'],
            dq2=input_data['dq2'], dk2=input_data['dk2'], dg=input_data['dg'], du=input_data['du'],
            cu_seqlens=input_data['cu_seqlens'] if input_data['cu_seqlens'].numel() > 0 else None,
            chunk_indices=input_data['chunk_indices'] if input_data['chunk_indices'].numel() > 0 else None,
            scale=input_data['scale'], T=input_data['T'], H=input_data['H'], K=input_data['K'],
            V=input_data['V'], BT=input_data['BT'], BK=input_data['BK'], BV=input_data['BV'],
        )
        torch_npu.npu.synchronize()

    # --- NPU Profiling ---
    prof_dir = f"prof_chunk_rwkv6_bwd_kernel_inter/shape_{T}_{H}_{K}_{V}_{BT}"

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
            chunk_rwkv6_bwd_kernel_inter[grid](
                q=input_data['q'], k=input_data['k'], v=input_data['v'], h=input_data['h'],
                gi=input_data['gi'], ge=input_data['ge'], u=input_data['u'], do=input_data['do'],
                dh=input_data['dh'], dA=input_data['dA'], dq=input_data['dq'], dk=input_data['dk'],
                dq2=input_data['dq2'], dk2=input_data['dk2'], dg=input_data['dg'], du=input_data['du'],
                cu_seqlens=input_data['cu_seqlens'] if input_data['cu_seqlens'].numel() > 0 else None,
                chunk_indices=input_data['chunk_indices'] if input_data['chunk_indices'].numel() > 0 else None,
                scale=input_data['scale'], T=input_data['T'], H=input_data['H'], K=input_data['K'],
                V=input_data['V'], BT=input_data['BT'], BK=input_data['BK'], BV=input_data['BV'],
            )
            torch_npu.npu.synchronize()
            prof.step()
        prof.stop()

    shape_elapsed = time.time() - shape_start
    print(f"Profile saved to: {prof_dir} (took {shape_elapsed:.1f}s)")


if __name__ == '__main__':
    test_chunk_rwkv6_bwd_kernel_inter_npu()
    prof_chunk_rwkv6_bwd_kernel_inter()
