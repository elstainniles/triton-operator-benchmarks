import math
import time
import torch
import torch_npu
from typing import Optional
import triton
import triton.language as tl


@triton.jit
def exp2(x): return tl.math.exp2(x.to(tl.float32))


@triton.jit(do_not_specialize=['T'])
def parallel_simple_gla_bwd_kernel_dq(
    i_t,
    i_k,
    i_v,
    q,
    k,
    v,
    g,
    do,
    dq,
    dg,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
):
    p_do = tl.make_block_ptr(do, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    # [BT, BV]
    b_do = tl.load(p_do, boundary_check=(0, 1))
    # [BT, BK]
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)

    # [BT]
    o_q = i_t * BT + tl.arange(0, BT)
    m_q = o_q < T
    for i_s in range(0, i_t * BT, BS):
        p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_s, i_k * BK), (BS, BK), (1, 0))
        p_v = tl.make_block_ptr(v, (V, T), (1, H*V), (i_v * BV, i_s), (BV, BS), (0, 1))

        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        # [BS, BK]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BV, BS]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BT, BV] @ [BV, BS] = [BT, BS]
        b_ds = tl.dot(b_do, b_v)
        if USE_G:
            b_g = tl.load(g + o_k * H, mask=m_k, other=0)
            b_gn = tl.load(g + (min(i_s + BS, T) - 1) * H)
            b_gp = tl.load(g + (i_s - 1) * H) if i_s % BT > 0 else 0.
            b_ds *= tl.where(m_k, exp2(b_gn - b_g), 0)[None, :]
            if i_s > 0:
                b_dq *= exp2(b_gn - b_gp)
        # [BT, BS] @ [BS, BK] = [BT, BK]
        b_dq += tl.dot(b_ds.to(b_v.dtype), b_k)

    if USE_G:
        # [BT,]
        b_gq = tl.load(g + o_q * H, mask=m_q, other=float('-inf'))
        # [BT, BK]
        b_dq *= exp2(b_gq)[:, None]
    # Q block and K block have overlap. masks required
    for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
        p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_s, i_k * BK), (BS, BK), (1, 0))
        p_v = tl.make_block_ptr(v, (V, T), (1, H*V), (i_v * BV, i_s), (BV, BS), (0, 1))

        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        # [BS, BK]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BV, BS]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BT, BV] @ [BV, BS] = [BT, BS]
        b_ds = tl.dot(b_do, b_v)
        if USE_G:
            b_gk = tl.load(g + o_k * H, mask=m_k, other=0)
            b_ds *= exp2(b_gq[:, None] - b_gk[None, :])
        m_s = (o_q[:, None] >= o_k[None, :]) & (m_q[:, None] & m_k[None, :])
        b_ds = tl.where(m_s, b_ds, 0)
        # [BT, BK]
        b_dq += tl.dot(b_ds.to(b_k.dtype), b_k)

    b_dq *= scale
    p_dq = tl.make_block_ptr(dq, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    if USE_G:
        p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_dg = tl.sum(b_dq * b_q, 1)
        p_dg = tl.make_block_ptr(dg, (T,), (H,), (i_t * BT,), (BT,), (0,))
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))


@triton.jit(do_not_specialize=['T'])
def parallel_simple_gla_bwd_kernel_dkv(
    i_t,
    i_k,
    i_v,
    q,
    k,
    v,
    g,
    do,
    dk,
    dv,
    dg,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
):
    o_k = i_t * BT + tl.arange(0, BT)
    m_k = o_k < T
    # [BT, BK]
    p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    # [BT, BV]
    p_v = tl.make_block_ptr(v, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_dv = tl.zeros([BT, BV], dtype=tl.float32)
    if USE_G:
        b_gk = tl.load(g + o_k * H, mask=m_k, other=0)
    NTS = tl.cdiv(T, BS)
    # [BT, BK]
    for i_s in range(NTS * BS - BS, (i_t + 1) * BT - BS, -BS):
        p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_s, i_k * BK), (BS, BK), (1, 0))
        p_do = tl.make_block_ptr(do, (T, V), (H*V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))

        o_q = i_s + tl.arange(0, BS)
        m_q = o_q < T
        # [BS, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BS, BV]
        b_do = tl.load(p_do, boundary_check=(0, 1))
        # [BT, BS]
        b_ds = tl.dot(b_v, tl.trans(b_do))
        b_s = tl.dot(b_k, tl.trans(b_q))
        if USE_G:
            b_gq = tl.load(g + o_q * H, mask=m_q, other=float('-inf'))
            b_gp = tl.load(g + (min(i_s + BS, T) - 1) * H)
            b_gn = tl.load(g + (i_s - 1) * H) if i_s % BT > 0 else 0.
            if i_s >= 0:
                b_gpn = exp2(b_gp - b_gn)
                b_dk *= b_gpn
                b_dv *= b_gpn
                b_gqn = exp2(b_gq - b_gn)
                b_ds *= b_gqn[None, :]
                b_s *= b_gqn[None, :]
        # [BT, BK]
        b_dk += tl.dot(b_ds.to(b_q.dtype), b_q)
        # [BT, BV]
        b_dv += tl.dot(b_s.to(b_do.dtype), b_do)

    if USE_G:
        b_gn = tl.load(g + (min(i_t * BT + BT, T) - 1) * H)
        if i_t >= 0:
            b_gpn = exp2(b_gn - b_gk)[:, None]
            b_dk *= b_gpn
            b_dv *= b_gpn

    for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
        p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_s, i_k * BK), (BS, BK), (1, 0))
        p_do = tl.make_block_ptr(do, (T, V), (H*V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))

        o_q = i_s + tl.arange(0, BS)
        m_q = o_q < T
        # [BS, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BS, BV]
        b_do = tl.load(p_do, boundary_check=(0, 1))
        # [BS]
        b_s = tl.dot(b_k, tl.trans(b_q))
        b_ds = tl.dot(b_v, tl.trans(b_do))
        if USE_G:
            b_gq = tl.load(g + o_q * H, mask=m_q, other=float('-inf'))
            if i_s >= 0:
                b_gkq = exp2(-b_gk[:, None] + b_gq[None, :])
                b_ds *= b_gkq
                b_s *= b_gkq
        m_s = o_k[:, None] <= o_q[None, :]
        b_s = tl.where(m_s, b_s, 0)
        b_ds = tl.where(m_s, b_ds, 0)
        # [BT, BK]
        b_dk += tl.dot(b_ds.to(b_q.dtype), b_q)
        b_dv += tl.dot(b_s.to(b_do.dtype), b_do)
    b_dk *= scale
    b_dv *= scale
    p_dk = tl.make_block_ptr(dk, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dv = tl.make_block_ptr(dv, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
    if USE_G:
        b_dg = tl.load(dg + o_k * H, mask=m_k, other=0)
        b_dg -= tl.sum(b_dk * b_k, 1)
        tl.store(dg + o_k * H, b_dg.to(dg.dtype.element_ty), mask=m_k)



@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    'NV': lambda args: triton.cdiv(args['V'], args['BV']),
})
@triton.jit(do_not_specialize=['T'])
def parallel_simple_gla_bwd_kernel(
    q,
    k,
    v,
    g,
    do,
    dq,
    dk,
    dv,
    dg,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_G: tl.constexpr,
):
    i_kv, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_k, i_v = i_kv // NV, i_kv % NV
    i_b, i_h = i_bh // H, i_bh % H
    dq += i_v * B * H * T * K
    dk += i_v * B * H * T * K
    dv += i_k * B * H * T * V
    if USE_G:
        dg += i_kv * B * H * T

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    do += (bos * H + i_h) * V
    dq += (bos * H + i_h) * K
    dk += (bos * H + i_h) * K
    dv += (bos * H + i_h) * V
    if USE_G:
        g += bos * H + i_h
        dg += bos * H + i_h

    parallel_simple_gla_bwd_kernel_dq(
        i_t=i_t,
        i_k=i_k,
        i_v=i_v,
        q=q,
        k=k,
        v=v,
        g=g,
        do=do,
        dq=dq,
        dg=dg,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BS=BS,
        BK=BK,
        BV=BV,
        USE_G=USE_G,
    )
    tl.debug_barrier()
    parallel_simple_gla_bwd_kernel_dkv(
        i_t=i_t,
        i_k=i_k,
        i_v=i_v,
        q=q,
        k=k,
        v=v,
        g=g,
        do=do,
        dk=dk,
        dv=dv,
        dg=dg,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BS=BS,
        BK=BK,
        BV=BV,
        USE_G=USE_G,
    )



def parallel_simple_gla_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    do: torch.Tensor,
    scale: float,
    chunk_size: int = 128,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
):
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT, BS = chunk_size, 32

    BK = min(32, triton.next_power_of_2(K))
    BV = min(32, triton.next_power_of_2(V))

    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    assert BT % BS == 0

    dq = torch.empty(NV, * q.shape, dtype=q.dtype if NV == 1 else torch.float, device=q.device)
    dk = torch.empty(NV, * k.shape, dtype=k.dtype if NV == 1 else torch.float, device=q.device)
    dv = torch.empty(NK, * v.shape, dtype=v.dtype if NK == 1 else torch.float, device=q.device)
    dg = torch.empty(NK*NV, *g.shape, dtype=torch.float, device=q.device) if g is not None else None

    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    grid = (NK * NV, NT, B * H)
    parallel_simple_gla_bwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        do=do,
        dq=dq,
        dk=dk,
        dv=dv,
        dg=dg,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        B=B,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BS=BS,
        BK=BK,
        BV=BV,
    )
    dq = dq.sum(0)
    dk = dk.sum(0)
    dv = dv.sum(0)
    dg = dg.sum(0)
    return dq, dk, dv, dg

def validate_cmp(dtype, y_cal, y_ref, overflow_mode=None, device_type=None):
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


def compare_data_precision(dict_ref, dict_cal, device_type):
    keys_ref, keys_cal = set(dict_ref.keys()), set(dict_cal.keys())
    if not keys_ref.issubset(keys_cal):
        raise ValueError("The keys of dict_ref is not subset of dict_cal")
    for key in dict_ref.keys():
        print("key:",key)
        val_a, val_b = dict_ref[key], dict_cal[key]
        if val_a is None and val_b is None:
            continue
        if type(val_a) != type(val_b):
            raise ValueError("The data type of two dicts are different")
        if isinstance(val_a, torch.Tensor):
            validate_cmp(dtype=str(val_a.dtype).split('.')[-1], y_ref=val_a, y_cal=val_b, device_type=device_type)
        else:
            raise ValueError("Non-tensor type is not currently supported")


def convert_tensor_with_device_type(indata, device_type):
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


def test_parallel_simple_gla_bwd():
    ptfile_path = "/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/parallel_simple_gla_bwd_kernel/test_parallel_simple_gla_bwd_cuda.pt"
    data = torch.load(ptfile_path, map_location=torch.device('cpu'), weights_only=False)

    device = torch.device('npu')
    input_data = convert_tensor_with_device_type(data["input_data"], device_type=device)

    q = input_data['q']
    k = input_data['k']
    v = input_data['v']
    g = input_data['g']
    do = input_data['do']
    scale = input_data['scale']
    chunk_size = input_data['chunk_size']
    cu_seqlens = input_data['cu_seqlens']
    chunk_indices = input_data['chunk_indices']
    print(f"{q.mean()=} {q.max()=} {q.min()=}")
    print(f"{k.mean()=} {k.max()=} {k.min()=}")
    print(f"{v.mean()=} {v.max()=} {v.min()=}")
    print(f"{g.mean()=} {g.max()=} {g.min()=}")
    print(f"{do.mean()=} {do.max()=} {do.min()=}")

    print(f"{data['gpu_output']['dq'].mean()=} {data['gpu_output']['dq'].max()=} {data['gpu_output']['dq'].min()=}")
    print(f"{data['gpu_output']['dk'].mean()=} {data['gpu_output']['dk'].max()=} {data['gpu_output']['dk'].min()=}")
    print(f"{data['gpu_output']['dv'].mean()=} {data['gpu_output']['dv'].max()=} {data['gpu_output']['dv'].min()=}")
    print(f"{data['gpu_output']['dg'].mean()=} {data['gpu_output']['dg'].max()=} {data['gpu_output']['dg'].min()=}")

    dq, dk, dv, dg = parallel_simple_gla_bwd(
        q=q,
        k=k,
        v=v,
        g=g,
        do=do,
        scale=scale,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )

    npu_out = {
        'dq': dq,
        'dk': dk,
        'dv': dv,
        'dg': dg,
    }

    try:
        compare_data_precision(data["gpu_output"], npu_out, device_type='cpu')
        print("Test parallel_simple_gla_bwd passed!")
    except (AssertionError, ValueError, RuntimeError) as e:
        print(f"Precision comparison failed: {e}")
        print("Falling back to fp64 third-party precision comparison...")
        fp64_path = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/parallel_simple_gla_bwd_kernel/test_parallel_simple_gla_bwd_cuda_fp64.pt'
        fp64_data = torch.load(fp64_path, map_location=torch.device('cpu'), weights_only=False)
        fp64_precision_compare(npu_out, data["gpu_output"], fp64_data["gpu_output"])


def prof_parallel_simple_gla_bwd():
    ptfile_path = "/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/parallel_simple_gla_bwd_kernel/test_parallel_simple_gla_bwd_cuda.pt"
    shape_start = time.time()
    data = torch.load(ptfile_path, map_location=torch.device('cpu'), weights_only=False)

    device = torch.device('npu')
    input_data = convert_tensor_with_device_type(data["input_data"], device_type=device)

    q = input_data['q']
    k = input_data['k']
    v = input_data['v']
    g = input_data['g']
    do = input_data['do']
    scale = input_data['scale']
    chunk_size = input_data['chunk_size']
    cu_seqlens = input_data['cu_seqlens']
    chunk_indices = input_data['chunk_indices']

    for _ in range(5):
        dq, dk, dv, dg = parallel_simple_gla_bwd(
            q=q, k=k, v=v, g=g, do=do, scale=scale,
            chunk_size=chunk_size, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        )
        torch_npu.npu.synchronize()

    B, T, H, K, V = *k.shape, v.shape[-1]
    prof_dir = f"prof_parallel_simple_gla_bwd/shape_{B}_{H}_{T}_{K}"

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
            dq, dk, dv, dg = parallel_simple_gla_bwd(
                q=q, k=k, v=v, g=g, do=do, scale=scale,
                chunk_size=chunk_size, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
            )
            torch_npu.npu.synchronize()
            prof.step()
        prof.stop()

    shape_elapsed = time.time() - shape_start
    print(f"Profile saved to: {prof_dir} (took {shape_elapsed:.1f}s)")


if __name__ == "__main__":
    test_parallel_simple_gla_bwd()
    prof_parallel_simple_gla_bwd()