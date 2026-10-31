import torch
import torch_npu
import torch.nn.functional as F
import triton
import triton.language as tl


def convert_tensor_with_device_type(tensor, device_type, dtype=None):
    if tensor is None:
        return None
    if dtype is not None:
        tensor = tensor.to(dtype)
    return tensor.to(device_type)


def compare_data_precision(output, ref_output, atol=1e-3, rtol=1e-3):
    if output is None and ref_output is None:
        return True
    if output is None or ref_output is None:
        return False
    if output.shape != ref_output.shape:
        return False
    return torch.allclose(output, ref_output, atol=atol, rtol=rtol)


def validate_cmp(output, ref_output, name, atol=1e-3, rtol=1e-3):
    if compare_data_precision(output, ref_output, atol=atol, rtol=rtol):
        print(f"[PASS] {name} matches reference")
        return True
    else:
        print(f"[FAIL] {name} does not match reference")
        if output is not None and ref_output is not None:
            diff = (output - ref_output).abs()
            print(f"  Max diff: {diff.max().item()}, Mean diff: {diff.mean().item()}")
        return False


FLA_CACHE_RESULTS = False
SUPPORTS_AUTOTUNE_CACHE = "cache_results" in triton.autotune.__code__.co_varnames
autotune_cache_kwargs = {"cache_results": FLA_CACHE_RESULTS} if SUPPORTS_AUTOTUNE_CACHE else {}


@triton.jit
def exp(x):
    return tl.exp(x.to(tl.float32))


def prepare_lens(cu_seqlens):
    return torch.diff(cu_seqlens)


def prepare_chunk_indices(cu_seqlens, chunk_size, cu_seqlens_cpu=None):
    if cu_seqlens_cpu is not None:
        indices = torch.cat([torch.arange(n, device=cu_seqlens.device)
                            for n in triton.cdiv(prepare_lens(cu_seqlens_cpu), chunk_size).tolist()])
        return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)
    indices = torch.cat([torch.arange(n) for n in triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


def prepare_chunk_offsets(cu_seqlens, chunk_size):
    return F.pad(triton.cdiv(prepare_lens(cu_seqlens), chunk_size), (1, 0), value=0).cumsum(-1)


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK, 'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64, 128]
        for BV in [32, 64, 128]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['BT', 'USE_G', 'USE_GK', 'USE_GV'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_fwd_kernel_h_parallel(
    k,
    v,
    h,
    g,
    gk,
    gv,
    h0,
    ht,
    cu_seqlens,
    chunk_indices,
    T,
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
    i_kv, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    NV = tl.cdiv(V, BV)
    i_k, i_v = i_kv // NV, i_kv % NV
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        bos, eos = i_b * T, i_b * T + T
        NT = tl.cdiv(T, BT)
        i_n, i_tg = i_b, i_b * NT + i_t
    i_nh = i_n * H + i_h

    p_k = tl.make_block_ptr(k + (bos*H + i_h) * K, (K, T), (1, H*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
    p_v = tl.make_block_ptr(v + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_h = tl.make_block_ptr(h + (i_tg * H + i_h) * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))

    if i_t == 0:
        if USE_INITIAL_STATE:
            p_h0 = tl.make_block_ptr(h0 + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
            b_h = tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)
        else:
            b_h = tl.zeros([BK, BV], dtype=tl.float32)
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))

    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_v = tl.load(p_v, boundary_check=(0, 1))

    last_idx = min(i_t * BT + BT, T) - 1
    if USE_G:
        b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
        p_g = g + bos*H + (i_t * BT + tl.arange(0, BT)) * H + i_h
        b_g = tl.load(p_g, mask=(i_t * BT + tl.arange(0, BT) < T), other=0.)
        b_v = (b_v * exp(b_g_last - b_g)[:, None]).to(b_v.dtype)

    if USE_GK:
        p_gk = tl.make_block_ptr(gk + (bos*H + i_h) * K, (K, T), (1, H*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_gk_last = gk + (bos + last_idx) * H*K + i_h * K + i_k * BK + tl.arange(0, BK)
        b_gk_last = tl.load(p_gk_last, mask=(i_k * BK + tl.arange(0, BK) < K), other=0.)

        b_gk = tl.load(p_gk, boundary_check=(0, 1))
        b_k = (b_k * exp(b_gk_last[:, None] - b_gk)).to(b_k.dtype)

    if USE_GV:
        p_gv = tl.make_block_ptr(gv + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_gv_last = gv + (bos + last_idx) * H*V + i_h * V + i_v * BV + tl.arange(0, BV)

        b_gv_last = tl.load(p_gv_last, mask=(i_v * BV + tl.arange(0, BV) < V), other=0.)
        b_gv = tl.load(p_gv, boundary_check=(0, 1))
        b_v = (b_v * exp(b_gv_last[None, :] - b_gv)).to(b_v.dtype)

    b_h = tl.dot(b_k, b_v)
    if i_t < NT - 1:
        p_h = tl.make_block_ptr(h + ((i_tg + 1) * H + i_h) * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))
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
        for BK in [32, 64, 128]
        for BV in [32, 64, 128]
        for num_warps in [2, 4, 8, 16]
        for num_stages in [2, 3]
    ],
    key=['BT', 'USE_G', 'USE_GK', 'USE_GV'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_fwd_kernel_h_reduction(
    h,
    g,
    gk,
    gv,
    kvt,
    ht,
    cu_seqlens,
    chunk_offsets,
    T,
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
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    for i_t in range(NT):
        p_h = tl.make_block_ptr(h + ((boh + i_t) * H + i_h) * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h += tl.load(p_h, boundary_check=(0, 1)).to(tl.float32)
        if i_t > 0:
            tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))

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

    if STORE_FINAL_STATE:
        p_kvt = tl.make_block_ptr(kvt + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        p_ht = tl.make_block_ptr(ht + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h += tl.load(p_kvt, boundary_check=(0, 1)).to(tl.float32)
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


def chunk_fwd_h(
    k, v, g, gk, gv, h0, output_final_state,
    states_in_fp32=False, cu_seqlens=None, chunk_size=64, chunk_indices=None,
):
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT)

    h = k.new_empty(B, NT, H, K, V, dtype=torch.float)
    ht = k.new_empty(N, H, K, V, dtype=torch.float) if output_final_state else None
    def grid(meta): return (triton.cdiv(K, meta['BK']) * triton.cdiv(V, meta['BV']), NT, B * H)
    chunk_fwd_kernel_h_parallel[grid](
        k=k, v=v, h=h, g=g, gk=gk, gv=gv, h0=h0, ht=ht,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        T=T, H=H, K=K, V=V, BT=BT,
        USE_G=g is not None, USE_GK=gk is not None, USE_GV=gv is not None,
    )
    kvt, ht = ht, (torch.empty_like(ht) if output_final_state else None)
    def grid(meta): return (triton.cdiv(K, meta['BK']), triton.cdiv(V, meta['BV']), N * H)
    chunk_fwd_kernel_h_reduction[grid](
        h=h, g=g, gk=gk, gv=gv, kvt=kvt, ht=ht,
        cu_seqlens=cu_seqlens, chunk_offsets=chunk_offsets,
        T=T, H=H, K=K, V=V, BT=BT,
        USE_G=g is not None, USE_GK=gk is not None, USE_GV=gv is not None,
    )
    h = h.to(k.dtype) if not states_in_fp32 else h
    return h, ht


def chunk_local_cumsum(g, chunk_size):
    B, T, H = g.shape
    NT = (T + chunk_size - 1) // chunk_size
    T_padded = NT * chunk_size
    g_padded = torch.nn.functional.pad(g, (0, 0, 0, T_padded - T))
    g_padded = g_padded.view(B, NT, chunk_size, H)
    g_cumsum = g_padded.cumsum(dim=2)
    g_cumsum = g_cumsum.view(B, T_padded, H)
    return g_cumsum[:, :T]


def test_chunk_fwd_kernel_h_reduction():
    data = torch.load(f'/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_h_parallel_chunk_fwd_kernel_h_reduction/test_chunk_fwd_kernel_h_reduction_cuda.pt')

    device = torch.device('npu:0')
    torch.npu.set_device(device)

    k = convert_tensor_with_device_type(data['input_data']['k'], device, torch.float16)
    v = convert_tensor_with_device_type(data['input_data']['v'], device, torch.float16)
    g_per_token = convert_tensor_with_device_type(data['input_data']['g_per_token'], device, torch.float32)
    h0 = data['input_data']['h0']
    output_final_state = data['input_data']['output_final_state']
    B = data['input_data']['B']
    T = data['input_data']['T']
    H = data['input_data']['H']
    K = data['input_data']['K']
    V = data['input_data']['V']
    BT = data['input_data']['BT']

    g_cumsum = chunk_local_cumsum(g_per_token, BT)

    h_out, ht_out = chunk_fwd_h(
        k=k, v=v, g=g_cumsum, gk=None, gv=None, h0=h0,
        output_final_state=output_final_state, states_in_fp32=True, chunk_size=BT,
    )

    ref_h = data['gpu_output']['h']
    ref_ht = data['gpu_output']['ht']

    validate_cmp(h_out.cpu(), ref_h, 'h', atol=1e-2, rtol=1e-2)
    if ref_ht is not None:
        validate_cmp(ht_out.cpu(), ref_ht, 'ht', atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    test_chunk_fwd_kernel_h_reduction()
