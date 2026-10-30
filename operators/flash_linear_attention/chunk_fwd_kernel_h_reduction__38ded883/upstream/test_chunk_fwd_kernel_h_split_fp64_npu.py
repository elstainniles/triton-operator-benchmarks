import torch
import triton
import triton.language as tl
import torch_npu


@triton.jit
def exp(x): return tl.exp(x.to(tl.float32))

@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK, 'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64]
        for BV in [32, 64]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3]
    ],
    key=['BT', 'USE_G', 'USE_GK', 'USE_GV'],
)
@triton.jit(do_not_specialize=['T'])
def chunk_fwd_kernel_h_split(
    k,
    v,
    g,
    gk,
    gv,
    hs,
    hr,
    h0,
    ht,
    cu_seqlens,
    split_indices,
    T,
    S: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_k, i_v, i_sh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_ss, i_h = i_sh // H, i_sh % H
    if IS_VARLEN:
        i_n, i_s = tl.load(split_indices + i_ss * 2).to(tl.int32), tl.load(split_indices + i_ss * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NS = tl.cdiv(T, S)
    else:
        NS = tl.cdiv(T, S)
        i_n, i_s = i_ss // NS, i_ss % NS
        bos, eos = i_n * T, i_n * T + T
    i_nh = i_n * H + i_h

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if i_s == 0:
        if USE_INITIAL_STATE:
            p_h0 = tl.make_block_ptr(h0 + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
            b_h += tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)
        p_hr = tl.make_block_ptr(hr + i_sh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_hr, b_h.to(p_hr.dtype.element_ty), boundary_check=(0, 1))
    for i_t in range(tl.cdiv(i_s * S, BT), tl.cdiv(min(i_s * S + S, T), BT)):
        p_k = tl.make_block_ptr(k + (bos*H + i_h) * K, (K, T), (1, H*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_v = tl.make_block_ptr(v + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        last_idx = min(i_t * BT + BT, T) - 1

        if USE_G:
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
            p_g = g + bos*H + (i_t * BT + tl.arange(0, BT)) * H + i_h
            b_h *= exp(b_g_last)
            b_g = tl.load(p_g, mask=(i_t * BT + tl.arange(0, BT) < T), other=0.)
            b_v = (b_v * exp(b_g_last - b_g)[:, None]).to(b_v.dtype)

        if USE_GK:
            p_gk = tl.make_block_ptr(gk + (bos*H + i_h) * K, (K, T), (1, H*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
            p_gk_last = gk + (bos + last_idx) * H*K + i_h * K + i_k * BK + tl.arange(0, BK)

            b_gk_last = tl.load(p_gk_last, mask=(i_k * BK + tl.arange(0, BK) < K), other=0.)
            b_h *= exp(b_gk_last)[:, None]

            b_gk = tl.load(p_gk, boundary_check=(0, 1))
            b_k = (b_k * exp(b_gk_last[:, None] - b_gk)).to(b_k.dtype)

        if USE_GV:
            p_gv = tl.make_block_ptr(gv + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_gv_last = gv + (bos + last_idx) * H*V + i_h * V + i_v * BV + tl.arange(0, BV)

            b_gv_last = tl.load(p_gv_last, mask=(i_v * BV + tl.arange(0, BV) < V), other=0.)
            b_h *= exp(b_gv_last)[None, :]

            b_gv = tl.load(p_gv, boundary_check=(0, 1))
            b_v = (b_v * exp(b_gv_last[None, :] - b_gv)).to(b_v.dtype)

        b_h += tl.dot(b_k, b_v)

    if NS > 1:
        p_hs = tl.make_block_ptr(hs + i_sh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_hs, b_h.to(p_hs.dtype.element_ty), boundary_check=(0, 1))
    elif STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK, 'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64]
        for BV in [32, 64]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['BT', 'USE_G', 'USE_GK', 'USE_GV'],
)
@triton.jit(do_not_specialize=['T'])
def chunk_fwd_kernel_h_reduction(
    g,
    gk,
    gv,
    hs,
    hr,
    ht,
    cu_seqlens,
    split_offsets,
    T,
    S: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GV: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NS = tl.cdiv(T, S)
        boh = tl.load(split_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NS = tl.cdiv(T, S)
        boh = i_n * NS

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    for i_s in range(1, NS):
        p_hs = tl.make_block_ptr(hs + ((boh + i_s-1) * H + i_h) * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        p_hr = tl.make_block_ptr(hr + ((boh + i_s) * H + i_h) * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h += tl.load(p_hs, boundary_check=(0, 1)).to(tl.float32)
        tl.store(p_hr, b_h.to(p_hr.dtype.element_ty), boundary_check=(0, 1))

        for i_t in range(tl.cdiv(i_s * S, BT), tl.cdiv(min(i_s * S + S, T), BT)):
            last_idx = min(i_t * BT + BT, T) - 1
            if USE_G:
                b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
                b_h *= exp(b_g_last)

            if USE_GK:
                p_gk_last = gk + (bos + last_idx) * H*K + i_h * K + i_k * BK + tl.arange(0, BK)
                b_gk_last = tl.load(p_gk_last, mask=(i_k * BK + tl.arange(0, BK) < K), other=0.)
                b_h *= exp(b_gk_last)[:, None]

            if USE_GV:
                p_gv_last = gv + (bos + last_idx) * H*V + i_h * V + i_v * BV + tl.arange(0, BV)
                b_gv_last = tl.load(p_gv_last, mask=(i_v * BV + tl.arange(0, BV) < V), other=0.)
                b_h *= exp(b_gv_last)[None, :]

    if NS > 1:
        if STORE_FINAL_STATE:
            p_hs = tl.make_block_ptr(hs + ((boh + NS-1) * H + i_h)*K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
            p_ht = tl.make_block_ptr(ht + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
            b_h += tl.load(p_hs, boundary_check=(0, 1)).to(tl.float32)
            tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


def chunk_fwd_h(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    gk: torch.Tensor,
    gv: torch.Tensor,
    h0: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
    split_offsets: torch.LongTensor | None = None,
    split_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
    split_size: int = 256,
    states_in_fp32: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    S, BT = split_size, chunk_size
    assert S % BT == 0, f"The `split_size` (got {S}) must be a multiple of `chunk_size` {BT}"
    if cu_seqlens is None:
        N = B
        NS = N * triton.cdiv(T, S)
    else:
        N = len(cu_seqlens) - 1
        NS = split_offsets[-1]

    hs = k.new_empty(NS, H, K, V, dtype=torch.float)
    hr = k.new_empty(NS, H, K, V, dtype=torch.float if states_in_fp32 else k.dtype)
    ht = k.new_empty(N, H, K, V, dtype=torch.float) if output_final_state else None
    def grid(meta): return (triton.cdiv(K, meta['BK']), triton.cdiv(V, meta['BV']), NS * H)
    chunk_fwd_kernel_h_split[grid](
        k=k,
        v=v,
        g=g,
        gk=gk,
        gv=gv,
        hs=hs,
        hr=hr,
        h0=h0,
        ht=ht,
        cu_seqlens=cu_seqlens,
        split_indices=split_indices,
        T=T,
        S=S,
        H=H,
        K=K,
        V=V,
        BT=BT,
        USE_G=g is not None,
        USE_GK=gk is not None,
        USE_GV=gv is not None,
    )
    def grid(meta): return (triton.cdiv(K, meta['BK']), triton.cdiv(V, meta['BV']), N * H)
    chunk_fwd_kernel_h_reduction[grid](
        g=g,
        gk=gk,
        gv=gv,
        hs=hs,
        hr=hr,
        ht=ht,
        cu_seqlens=cu_seqlens,
        split_offsets=split_offsets,
        T=T,
        S=S,
        H=H,
        K=K,
        V=V,
        BT=BT,
        USE_G=g is not None,
        USE_GK=gk is not None,
        USE_GV=gv is not None,
    )
    return hr, ht


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


def cal_precision(input: torch.tensor, golden: torch.tensor):
    relative_max = torch.max(torch.abs(input-golden)/(torch.abs(golden)+float(1e-7)))
    print(f"max relative error: {relative_max}")
    relative_mean = torch.mean(torch.abs(input-golden)/(torch.abs(golden)+float(1e-7)))
    print(f"mean relative error: {relative_mean}")
    rmse = torch.sqrt(torch.mean((input-golden)**2))
    print(f"rmse: {rmse}")
    mask = (torch.abs(golden) < float(2e-14) ) & (torch.abs(input-golden) > float(2e-30))
    error = torch.abs(input-golden)
    total_error = torch.sum(error[mask])
    print(f"small val error: {total_error}")
    return relative_max, relative_mean, rmse


def test_chunk_fwd_kernel_h_split():
    ptfile_path = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_fwd_kernel_h_split_fp64/test_chunk_fwd_kernel_h_split_v2.pt'
    ptfile_path1 = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_fwd_kernel_h_split_fp64/test_chunk_fwd_kernel_h_split_fp64_v2.pt'
    data = torch.load(ptfile_path, map_location=torch.device('cpu'), weights_only=False)
    datafp64 = torch.load(ptfile_path1, map_location=torch.device('cpu'), weights_only=False)
    
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    
    chunk_size = input_data['chunk_size']
    output_final_state = input_data['output_final_state']
    g = input_data['g']
    gk = input_data['gk']
    gv = input_data['gv']
    h0 = input_data['h0']
    k = input_data['k']
    split_size = input_data['split_size']
    v = input_data['v']
    hr, ht = chunk_fwd_h(k=k, v=v, g=g, gk=gk, gv=gv, h0=h0, output_final_state=output_final_state, chunk_size=chunk_size, split_size=split_size)

    npu_out = {
        'hr': hr.cpu(),
        'ht': ht.cpu() if ht is not None else None,
    }
    
    # Compare with GPU fp64 output
    for key in datafp64['gpu_output']:
        if key in npu_out:
            gpu_output = data['gpu_output'][key]
            gpu_output_fp64 = datafp64['gpu_output'][key]
            npu_output = npu_out[key]
            print(f"Comparing {key}:")
            # cal_precision(npu_output, gpu_output_fp64)
            relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(npu_output, gpu_output_fp64)
            relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(gpu_output, gpu_output_fp64)
            print(
                f"最大相对误差比 :{relative_max_npu / relative_max_gpu}, 平均相对误差比 :{relative_mean_npu / relative_mean_gpu}, 均方根误差比 :{rmse_npu / rmse_gpu}")
            assert relative_max_npu / relative_max_gpu <= 2, "最大相对误差比 check failed"
            assert relative_mean_npu / relative_mean_gpu <= 1.2, "平均相对误差比 check failed"
            assert rmse_npu / rmse_gpu <= 1.2, "均方根误差比 check failed"
    print("Output comparison passed!")



if __name__ == '__main__':
    test_chunk_fwd_kernel_h_split()
