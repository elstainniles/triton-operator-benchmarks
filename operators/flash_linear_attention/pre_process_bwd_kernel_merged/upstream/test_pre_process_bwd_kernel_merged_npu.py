import torch
import torch_npu
from typing import Optional
import triton
import triton.language as tl


USE_CUDA_GRAPH = False
autotune_cache_kwargs = {}


def check_shared_mem(arch="none", tensor_idx=0):
    return False


@triton.jit
def exp2(x):
    return tl.math.exp2(x.to(tl.float32))


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_GK': lambda args: args['gk'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in ([4, 3, 2] if check_shared_mem('ampere') else [1])
    ],
    key=['H', 'HV', 'K', 'V', 'BT'],
    use_cuda_graph=USE_CUDA_GRAPH,
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def pre_process_bwd_kernel_merged(
    q,
    k,
    w,
    g,
    gk,
    do,
    dhm,
    dv,
    cu_seqlens,
    scale,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BK1: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_BG: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_col, i_h = tl.program_id(0), tl.program_id(1)
    i_n = 0
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
        NT = tl.cdiv(T, BT)
    else:
        bos, eos = (i_n * T).to(tl.int64), (i_n * T + T).to(tl.int64)
        NT = tl.cdiv(T, BT)

    is_dh_part = i_col * BLOCK_SIZE < V

    q += ((bos * H + i_h // (HV // H)) * K).to(tl.int64)
    k += ((bos * H + i_h // (HV // H)) * K).to(tl.int64)
    if USE_BG:
        w += ((bos * H + i_h // (HV // H)) * K).to(tl.int64)
    else:
        w += ((bos * HV + i_h) * K).to(tl.int64)
    if USE_G:
        g += (bos * HV + i_h).to(tl.int64)
    if USE_GK:
        gk += ((bos * HV + i_h) * K).to(tl.int64)
    dhm += i_h * K * (V + K)
    stride_qk = H * K
    stride_w = H * K if USE_BG else HV * K

    if is_dh_part:
        do += ((bos * HV + i_h) * V).to(tl.int64)
        dv += ((bos * HV + i_h) * V).to(tl.int64)
        stride_v = HV * V
        i_v = i_col

        b_dh1 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)
        if K > 64:
            b_dh2 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)
        if K > 128:
            b_dh3 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)
        if K > 192:
            b_dh4 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)

        for i_t in range(NT - 1, -1, -1):
            last_idx = min((i_t + 1) * BT, T) - 1

            if USE_G:
                bg_last = tl.load(g + last_idx * HV).to(tl.float32)
                p_g = tl.make_block_ptr(g, (T,), (HV,), (i_t * BT,), (BT,), (0,))
                b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
                bg_last_exp = exp2(bg_last)
                b_g_exp = exp2(b_g)

            p_dv = tl.make_block_ptr(dv, (T, V), (stride_v, 1), (i_t * BT, i_v * BLOCK_SIZE), (BT, BLOCK_SIZE), (1, 0))
            p_do = tl.make_block_ptr(do, (T, V), (stride_v, 1), (i_t * BT, i_v * BLOCK_SIZE), (BT, BLOCK_SIZE), (1, 0))
            b_do = tl.load(p_do, boundary_check=(0, 1))

            p_k = tl.make_block_ptr(k, (T, K), (stride_qk, 1), (i_t * BT, 0), (BT, 64), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if USE_GK:
                o_k1 = tl.arange(0, 64)
                p_gk_last = gk + last_idx * HV * K
                b_gk_last1 = tl.load(p_gk_last + o_k1, mask=(o_k1 < K), other=0.).to(tl.float32)
            b_dv = tl.dot(b_k, b_dh1.to(b_k.dtype))

            if K > 64:
                p_k = tl.make_block_ptr(k, (T, K), (stride_qk, 1), (i_t * BT, 64), (BT, 64), (1, 0))
                b_k = tl.load(p_k, boundary_check=(0, 1))
                if USE_GK:
                    o_k2 = 64 + o_k1
                    b_gk_last2 = tl.load(p_gk_last + o_k2, mask=(o_k2 < K), other=0.).to(tl.float32)
                b_dv += tl.dot(b_k, b_dh2.to(b_k.dtype))

            if K > 128:
                p_k = tl.make_block_ptr(k, (T, K), (stride_qk, 1), (i_t * BT, 128), (BT, 64), (1, 0))
                b_k = tl.load(p_k, boundary_check=(0, 1))
                if USE_GK:
                    o_k3 = 128 + o_k1
                    b_gk_last3 = tl.load(p_gk_last + o_k3, mask=(o_k3 < K), other=0.).to(tl.float32)
                b_dv += tl.dot(b_k, b_dh3.to(b_k.dtype))

            if K > 192:
                p_k = tl.make_block_ptr(k, (T, K), (stride_qk, 1), (i_t * BT, 192), (BT, 64), (1, 0))
                b_k = tl.load(p_k, boundary_check=(0, 1))
                if USE_GK:
                    o_k4 = 192 + o_k1
                    b_gk_last4 = tl.load(p_gk_last + o_k4, mask=(o_k4 < K), other=0.).to(tl.float32)
                b_dv += tl.dot(b_k, b_dh4.to(b_k.dtype))

            if USE_G:
                m_t = (i_t * BT + tl.arange(0, BT)) < T
                b_dv *= tl.where(m_t, exp2(bg_last - b_g), 0)[:, None]
            b_dv += tl.load(p_dv, boundary_check=(0, 1))

            p_w = tl.make_block_ptr(w, (K, T), (1, stride_w), (0, i_t * BT), (64, BT), (0, 1))
            p_q = tl.make_block_ptr(q, (K, T), (1, stride_qk), (0, i_t * BT), (64, BT), (0, 1))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            if USE_G:
                b_dh1 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            if USE_GK:
                b_dh1 *= exp2(b_gk_last1[:, None])
            if USE_BG:
                b_dh1 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) + tl.dot(b_w, b_dv.to(b_w.dtype))
            else:
                b_dh1 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))

            if K > 64:
                p_q = tl.make_block_ptr(q, (K, T), (1, stride_qk), (64, i_t * BT), (64, BT), (0, 1))
                p_w = tl.make_block_ptr(w, (K, T), (1, stride_w), (64, i_t * BT), (64, BT), (0, 1))
                b_q = tl.load(p_q, boundary_check=(0, 1))
                b_w = tl.load(p_w, boundary_check=(0, 1))
                if USE_G:
                    b_dh2 *= bg_last_exp
                    b_q = b_q * b_g_exp[None, :]
                if USE_GK:
                    b_dh2 *= exp2(b_gk_last2[:, None])
                if USE_BG:
                    b_dh2 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) + tl.dot(b_w, b_dv.to(b_w.dtype))
                else:
                    b_dh2 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))

            if K > 128:
                p_q = tl.make_block_ptr(q, (K, T), (1, stride_qk), (128, i_t * BT), (64, BT), (0, 1))
                p_w = tl.make_block_ptr(w, (K, T), (1, stride_w), (128, i_t * BT), (64, BT), (0, 1))
                b_q = tl.load(p_q, boundary_check=(0, 1))
                b_w = tl.load(p_w, boundary_check=(0, 1))
                if USE_G:
                    b_dh3 *= bg_last_exp
                    b_q = b_q * b_g_exp[None, :]
                if USE_GK:
                    b_dh3 *= exp2(b_gk_last3[:, None])
                if USE_BG:
                    b_dh3 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) + tl.dot(b_w, b_dv.to(b_w.dtype))
                else:
                    b_dh3 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))

            if K > 192:
                p_q = tl.make_block_ptr(q, (K, T), (1, stride_qk), (192, i_t * BT), (64, BT), (0, 1))
                p_w = tl.make_block_ptr(w, (K, T), (1, stride_w), (192, i_t * BT), (64, BT), (0, 1))
                b_q = tl.load(p_q, boundary_check=(0, 1))
                b_w = tl.load(p_w, boundary_check=(0, 1))
                if USE_G:
                    b_dh4 *= bg_last_exp
                    b_q = b_q * b_g_exp[None, :]
                if USE_GK:
                    b_dh4 *= exp2(b_gk_last4[:, None])
                if USE_BG:
                    b_dh4 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) + tl.dot(b_w, b_dv.to(b_w.dtype))
                else:
                    b_dh4 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))

        p_dh1 = tl.make_block_ptr(dhm, (K, V), (V + K, 1), (0, i_v * BLOCK_SIZE), (64, BLOCK_SIZE), (1, 0))
        tl.store(p_dh1, b_dh1.to(p_dh1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_dh2 = tl.make_block_ptr(dhm, (K, V), (V + K, 1), (64, i_v * BLOCK_SIZE), (64, BLOCK_SIZE), (1, 0))
            tl.store(p_dh2, b_dh2.to(p_dh2.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_dh3 = tl.make_block_ptr(dhm, (K, V), (V + K, 1), (128, i_v * BLOCK_SIZE), (64, BLOCK_SIZE), (1, 0))
            tl.store(p_dh3, b_dh3.to(p_dh3.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_dh4 = tl.make_block_ptr(dhm, (K, V), (V + K, 1), (192, i_v * BLOCK_SIZE), (64, BLOCK_SIZE), (1, 0))
            tl.store(p_dh4, b_dh4.to(p_dh4.dtype.element_ty), boundary_check=(0, 1))
    else:
        i_k_col = i_col - tl.cdiv(V, BLOCK_SIZE)

        row = tl.arange(0, BK1)
        col = tl.arange(0, BLOCK_SIZE) + i_k_col * BLOCK_SIZE

        b_m = tl.where(row[:, None] == col[None, :], 1.0, 0.0)

        for _i_t in range(NT):
            i_t = NT - 1 - _i_t

            p_k = tl.make_block_ptr(k, (T, K), (stride_qk, 1), (i_t * BT, 0), (BT, BK1), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            p_w = tl.make_block_ptr(w, (T, K), (stride_w, 1), (i_t * BT, 0), (BT, BK1), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))

            last_idx = min((i_t + 1) * BT, T) - 1

            if USE_G:
                m_t = (i_t * BT + tl.arange(0, BT)) < T
                b_g_last = tl.load(g + last_idx * HV).to(tl.float32)
                p_g = tl.make_block_ptr(g, (T,), (HV,), (i_t * BT,), (BT,), (0,))
                b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
                b_k = b_k * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
                b_g_last = exp2(b_g_last)
                b_diag = tl.where(row[:, None] == row[None, :], b_g_last, 0.0)
            elif USE_GK:
                b_gk_last = tl.load(gk + last_idx * HV * K + row, mask=(row < K), other=0.).to(tl.float32)
                b_gk_last = exp2(b_gk_last)
                b_diag = tl.where(row[:, None] == row[None, :], b_gk_last[:, None], 0.0)
            else:
                b_diag = tl.where(row[:, None] == row[None, :], 1.0, 0.0)

            b_kw = tl.dot(tl.trans(b_w), b_k.to(b_w.dtype))
            if USE_BG:
                b_m_i = b_diag + b_kw
            else:
                b_m_i = b_diag - b_kw
            b_m = tl.dot(b_m_i.to(tl.float32), b_m.to(tl.float32))

        p_m = tl.make_block_ptr(dhm + V, (K, K), (V + K, 1), (0, i_k_col * BLOCK_SIZE), (BK1, BLOCK_SIZE), (1, 0))
        tl.store(p_m, b_m.to(p_m.dtype.element_ty), boundary_check=(0, 1))


def chunk_gated_delta_rule_bwd_dhu_pre_process(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    do: torch.Tensor,
    dv: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    bg: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
):
    B, T, H, K, V, HV = *q.shape, do.shape[-1], do.shape[2]
    BT = chunk_size
    assert K <= 256, "current kernel does not support head dimension being larger than 256."
    BK = triton.next_power_of_2(K)

    if cu_seqlens is None:
        N = B
    else:
        N = len(cu_seqlens) - 1

    dhm = q.new_zeros(HV, K, V + K, dtype=torch.float32)

    BLOCK_SIZE = 32 if K <= 64 else 64
    grid = (triton.cdiv(V, BLOCK_SIZE) + triton.cdiv(K, BLOCK_SIZE), HV)
    pre_process_bwd_kernel_merged[grid](
        q=q,
        k=k if bg is None else bg,
        w=w,
        g=g,
        gk=gk,
        do=do,
        dhm=dhm,
        dv=dv,
        cu_seqlens=cu_seqlens,
        scale=scale,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        BK1=BK,
        BLOCK_SIZE=BLOCK_SIZE,
        USE_BG=bg is not None,
    )
    return dhm

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

def test_pre_process_bwd_kernel_merged():
    ptfile_path = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/test_pre_process_bwd_kernel_merged/test_pre_process_bwd_kernel_merged_cuda.pt'
    data = torch.load(ptfile_path, map_location=torch.device('cpu'), weights_only=False)

    device = torch.device('npu')
    input_data = convert_tensor_with_device_type(data["input_data"], device_type='npu')

    dhm = chunk_gated_delta_rule_bwd_dhu_pre_process(
        input_data['q'], input_data['k'], input_data['w'],
        input_data['do'], input_data['dv'],
        g=input_data['g'], gk=input_data['gk'], bg=input_data['bg'],
        scale=input_data['scale'],
        cu_seqlens=input_data['cu_seqlens'],
        chunk_size=input_data['chunk_size'],
    )

    npu_out = {
        'dhm': dhm,
    }

    compare_data_precision(data["gpu_output"], npu_out, device_type='cpu')
    print("Test pre_process_bwd_kernel_merged passed!")


if __name__ == "__main__":
    test_pre_process_bwd_kernel_merged()
