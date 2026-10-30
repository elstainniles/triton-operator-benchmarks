# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import os
from typing import Optional
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import torch_npu


if os.environ.get('FLA_USE_FAST_OPS', '0') == '1':
    @triton.jit
    def exp2(x): return tl.math.exp2(x.to(tl.float32))
else:
    @triton.jit
    def exp2(x): return tl.math.exp2(x.to(tl.float32))


def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return torch.diff(cu_seqlens)


def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
    cu_seqlens_cpu: torch.LongTensor | None = None,
) -> torch.LongTensor:
    if cu_seqlens_cpu is not None:
        indices = torch.cat([torch.arange(n, device=cu_seqlens.device)
                            for n in triton.cdiv(prepare_lens(cu_seqlens_cpu), chunk_size).tolist()])
        return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)
    indices = torch.cat([torch.arange(n) for n in triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


def prepare_chunk_offsets(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
) -> torch.LongTensor:
    return F.pad(triton.cdiv(prepare_lens(cu_seqlens), chunk_size), (1, 0), value=0).cumsum(-1)


def check_shared_mem(arch: str = "none", tensor_idx: int = 0) -> bool:
    return False


def fla_cache_autotune(configs, key=None, prune_configs_by=None, reset_to_zero=None, restore_value=None,
                       pre_hook=None, post_hook=None, warmup=None, rep=None, use_cuda_graph=False,
                       do_bench=None, cache_results=False):
    if key is None:
        key = []
    
    def decorator(fn):
        return triton.autotune(
            configs=configs,
            key=key,
            reset_to_zero=reset_to_zero,
            restore_value=restore_value,
            pre_hook=pre_hook,
            post_hook=post_hook,
            warmup=warmup,
            rep=rep,
        )(fn)
    
    return decorator


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


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_GK': lambda args: args['gk'] is not None,
    'USE_INITIAL_STATE': lambda args: args['dh0'] is not None,
    'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@fla_cache_autotune(
    configs=[
        triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in ([2, 3, 4] if check_shared_mem('ampere') else [1])
        for BV in ([32, 64] if check_shared_mem('ada') else [32])
    ],
    key=['H', 'HV', 'K', 'V', 'BT', 'BV', 'USE_G', 'TRANSPOSE_STATE'],
    use_cuda_graph=False,
)
@triton.jit(do_not_specialize=['T'])
def chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64(
    q,
    k,
    w,
    g,
    gk,
    dht,
    dh0,
    do,
    dh,
    dv,
    dv2,
    cu_seqlens,
    chunk_offsets,
    scale,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    TRANSPOSE_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // HV, i_nh % HV
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    if TRANSPOSE_STATE:
        b_dh1 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 64:
            b_dh2 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 128:
            b_dh3 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 192:
            b_dh4 = tl.zeros([BV, 64], dtype=tl.float32)
    else:
        b_dh1 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 64:
            b_dh2 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 128:
            b_dh3 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 192:
            b_dh4 = tl.zeros([64, BV], dtype=tl.float32)

    q += (bos * H + i_h // (HV // H)).to(tl.int64) * K
    k += (bos * H + i_h // (HV // H)).to(tl.int64) * K
    w += (bos * HV + i_h).to(tl.int64) * K
    do += (bos * HV + i_h).to(tl.int64) * V
    dv += (bos * HV + i_h).to(tl.int64) * V
    dv2 += (bos * HV + i_h).to(tl.int64) * V
    dh += (boh * HV + i_h).to(tl.int64) * K*V
    if USE_GK:
        gk += (bos * HV + i_h).to(tl.int64) * K

    if USE_INITIAL_STATE:
        dh0 += i_nh * K*V
    if USE_FINAL_STATE_GRADIENT:
        dht += i_nh * K*V

    if USE_FINAL_STATE_GRADIENT:
        if TRANSPOSE_STATE:
            p_dht1 = tl.make_block_ptr(dht, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        else:
            p_dht1 = tl.make_block_ptr(dht, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        b_dh1 += tl.load(p_dht1, boundary_check=(0, 1))
        if K > 64:
            if TRANSPOSE_STATE:
                p_dht2 = tl.make_block_ptr(dht, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
            else:
                p_dht2 = tl.make_block_ptr(dht, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            b_dh2 += tl.load(p_dht2, boundary_check=(0, 1))
        if K > 128:
            if TRANSPOSE_STATE:
                p_dht3 = tl.make_block_ptr(dht, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0))
            else:
                p_dht3 = tl.make_block_ptr(dht, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            b_dh3 += tl.load(p_dht3, boundary_check=(0, 1))
        if K > 192:
            if TRANSPOSE_STATE:
                p_dht4 = tl.make_block_ptr(dht, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0))
            else:
                p_dht4 = tl.make_block_ptr(dht, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            b_dh4 += tl.load(p_dht4, boundary_check=(0, 1))

    for i_t in range(NT - 1, -1, -1):
        i_t_int64 = i_t.to(tl.int64)
        if TRANSPOSE_STATE:
            p_dh1 = tl.make_block_ptr(dh + i_t_int64*HV*K*V, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        else:
            p_dh1 = tl.make_block_ptr(dh + i_t_int64*HV*K*V, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_dh1, b_dh1.to(p_dh1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            if TRANSPOSE_STATE:
                p_dh2 = tl.make_block_ptr(dh + i_t_int64*HV*K*V, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
            else:
                p_dh2 = tl.make_block_ptr(dh + i_t_int64*HV*K*V, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh2, b_dh2.to(p_dh2.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            if TRANSPOSE_STATE:
                p_dh3 = tl.make_block_ptr(dh + i_t_int64*HV*K*V, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0))
            else:
                p_dh3 = tl.make_block_ptr(dh + i_t_int64*HV*K*V, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh3, b_dh3.to(p_dh3.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            if TRANSPOSE_STATE:
                p_dh4 = tl.make_block_ptr(dh + i_t_int64*HV*K*V, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0))
            else:
                p_dh4 = tl.make_block_ptr(dh + i_t_int64*HV*K*V, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh4, b_dh4.to(p_dh4.dtype.element_ty), boundary_check=(0, 1))

        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            bg_last = tl.load(g + (bos + last_idx) * HV + i_h).to(tl.float32)
            p_g = tl.make_block_ptr(g + bos * HV + i_h, (T,), (HV,), (i_t * BT,), (BT,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
            bg_last_exp = exp2(bg_last)
            b_g_exp = exp2(b_g)
        p_dv = tl.make_block_ptr(dv, (T, V), (HV*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dv2 = tl.make_block_ptr(dv2, (T, V), (HV*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_do = tl.make_block_ptr(do, (T, V), (HV*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

        b_do = tl.load(p_do, boundary_check=(0, 1))

        p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT, 0), (BT, 64), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        if USE_GK:
            o_k1 = tl.arange(0, 64)
            b_gk_last1 = tl.load(gk + last_idx * HV*K + o_k1, mask=(o_k1 < K), other=0.).to(tl.float32)
        if TRANSPOSE_STATE:
            b_dv = tl.dot(b_k, tl.trans(b_dh1).to(b_k.dtype))
        else:
            b_dv = tl.dot(b_k, b_dh1.to(b_k.dtype))

        if K > 64:
            p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT, 64), (BT, 64), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if USE_GK:
                o_k2 = 64 + o_k1
                b_gk_last2 = tl.load(gk + last_idx * HV*K + o_k2, mask=(o_k2 < K), other=0.).to(tl.float32)
            if TRANSPOSE_STATE:
                b_dv += tl.dot(b_k, tl.trans(b_dh2).to(b_k.dtype))
            else:
                b_dv += tl.dot(b_k, b_dh2.to(b_k.dtype))

        if K > 128:
            p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT, 128), (BT, 64), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if USE_GK:
                o_k3 = 128 + o_k1
                b_gk_last3 = tl.load(gk + last_idx * HV*K + o_k3, mask=(o_k3 < K), other=0.).to(tl.float32)
            if TRANSPOSE_STATE:
                b_dv += tl.dot(b_k, tl.trans(b_dh3).to(b_k.dtype))
            else:
                b_dv += tl.dot(b_k, b_dh3.to(b_k.dtype))

        if K > 192:
            p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT, 192), (BT, 64), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if USE_GK:
                o_k4 = 192 + o_k1
                b_gk_last4 = tl.load(gk + last_idx * HV*K + o_k4, mask=(o_k4 < K), other=0.).to(tl.float32)
            if TRANSPOSE_STATE:
                b_dv += tl.dot(b_k, tl.trans(b_dh4).to(b_k.dtype))
            else:
                b_dv += tl.dot(b_k, b_dh4.to(b_k.dtype))

        if USE_G:
            m_t = (i_t * BT + tl.arange(0, BT)) < T
            b_dv *= tl.where(m_t, exp2(bg_last - b_g), 0)[:, None]
        b_dv += tl.load(p_dv, boundary_check=(0, 1))

        tl.store(p_dv2, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
        p_w = tl.make_block_ptr(w, (K, T), (1, HV*K), (0, i_t * BT), (64, BT), (0, 1))
        p_q = tl.make_block_ptr(q, (K, T), (1, H*K), (0, i_t * BT), (64, BT), (0, 1))
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        if USE_G:
            b_dh1 *= bg_last_exp
            b_q = b_q * b_g_exp[None, :]
        if USE_GK:
            if TRANSPOSE_STATE:
                b_dh1 *= exp2(b_gk_last1)[None, :]
            else:
                b_dh1 *= exp2(b_gk_last1[:, None])
        if TRANSPOSE_STATE:
            b_dh1 += tl.trans(tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype)))
        else:
            b_dh1 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))
        if K > 64:
            p_q = tl.make_block_ptr(q, (K, T), (1, H*K), (64, i_t * BT), (64, BT), (0, 1))
            p_w = tl.make_block_ptr(w, (K, T), (1, HV*K), (64, i_t * BT), (64, BT), (0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if USE_G:
                b_dh2 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            if USE_GK:
                if TRANSPOSE_STATE:
                    b_dh2 *= exp2(b_gk_last2)[None, :]
                else:
                    b_dh2 *= exp2(b_gk_last2[:, None])
            if TRANSPOSE_STATE:
                b_dh2 += tl.trans(tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype)))
            else:
                b_dh2 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))
        if K > 128:
            p_q = tl.make_block_ptr(q, (K, T), (1, H*K), (128, i_t * BT), (64, BT), (0, 1))
            p_w = tl.make_block_ptr(w, (K, T), (1, HV*K), (128, i_t * BT), (64, BT), (0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if USE_G:
                b_dh3 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            if USE_GK:
                if TRANSPOSE_STATE:
                    b_dh3 *= exp2(b_gk_last3)[None, :]
                else:
                    b_dh3 *= exp2(b_gk_last3[:, None])
            if TRANSPOSE_STATE:
                b_dh3 += tl.trans(tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype)))
            else:
                b_dh3 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))
        if K > 192:
            p_q = tl.make_block_ptr(q, (K, T), (1, H*K), (192, i_t * BT), (64, BT), (0, 1))
            p_w = tl.make_block_ptr(w, (K, T), (1, HV*K), (192, i_t * BT), (64, BT), (0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if USE_G:
                b_dh4 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            if USE_GK:
                if TRANSPOSE_STATE:
                    b_dh4 *= exp2(b_gk_last4)[None, :]
                else:
                    b_dh4 *= exp2(b_gk_last4[:, None])
            if TRANSPOSE_STATE:
                b_dh4 += tl.trans(tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype)))
            else:
                b_dh4 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))

    if USE_INITIAL_STATE:
        if TRANSPOSE_STATE:
            p_dh0 = tl.make_block_ptr(dh0, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        else:
            p_dh0 = tl.make_block_ptr(dh0, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_dh0, b_dh1.to(p_dh0.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            if TRANSPOSE_STATE:
                p_dh1 = tl.make_block_ptr(dh0, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
            else:
                p_dh1 = tl.make_block_ptr(dh0, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh1, b_dh2.to(p_dh1.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            if TRANSPOSE_STATE:
                p_dh2 = tl.make_block_ptr(dh0, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0))
            else:
                p_dh2 = tl.make_block_ptr(dh0, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh2, b_dh3.to(p_dh2.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            if TRANSPOSE_STATE:
                p_dh3 = tl.make_block_ptr(dh0, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0))
            else:
                p_dh3 = tl.make_block_ptr(dh0, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh3, b_dh4.to(p_dh3.dtype.element_ty), boundary_check=(0, 1))


def chunk_gated_delta_rule_bwd_dhu(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    do: torch.Tensor,
    dv: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    h0: torch.Tensor | None = None,
    dht: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
    transpose_state_layout: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V, HV = *q.shape, do.shape[-1], do.shape[2]
    BT = chunk_size
    assert K <= 256, "current kernel does not support head dimension being larger than 256."

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT)

    if transpose_state_layout:
        dh = q.new_empty(B, NT, HV, V, K)
    else:
        dh = q.new_empty(B, NT, HV, K, V)
    dh0 = torch.empty_like(h0, dtype=torch.float) if h0 is not None else None
    dv2 = torch.empty_like(dv)

    def grid(meta): return (triton.cdiv(V, meta['BV']), N*HV)
    chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64[grid](
        q=q,
        k=k,
        w=w,
        g=g,
        gk=gk,
        dht=dht,
        dh0=dh0,
        do=do,
        dh=dh,
        dv=dv,
        dv2=dv2,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        scale=scale,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        TRANSPOSE_STATE=transpose_state_layout,
    )
    return dh, dh0, dv2


def cal_precision(input: torch.tensor, golden: torch.tensor):
    valid_mask = ~(torch.isnan(input) | torch.isnan(golden) | torch.isinf(input) | torch.isinf(golden))
    valid_count = valid_mask.sum().item()
    total_count = input.numel()
    nan_count_input = torch.isnan(input).sum().item()
    nan_count_golden = torch.isnan(golden).sum().item()
    inf_count_input = torch.isinf(input).sum().item()
    inf_count_golden = torch.isinf(golden).sum().item()
    print(f"Total elements: {total_count}, Valid elements: {valid_count}")
    print(f"  Input - NaN: {nan_count_input}, Inf: {inf_count_input}")
    print(f"  Golden - NaN: {nan_count_golden}, Inf: {inf_count_golden}")
    
    if valid_count == 0:
        print("WARNING: All values are NaN or Inf, returning inf")
        return float('inf'), float('inf'), float('inf')
    
    input_valid = input[valid_mask]
    golden_valid = golden[valid_mask]
    
    diff = torch.abs(input_valid - golden_valid)
    golden_abs = torch.abs(golden_valid) + float(1e-7)
    relative_errors = diff / golden_abs
    
    relative_max = torch.max(relative_errors).item()
    print(f"max relative error: {relative_max}")
    relative_mean = torch.mean(relative_errors).item()
    print(f"mean relative error: {relative_mean}")
    rmse = torch.sqrt(torch.mean(diff ** 2)).item()
    print(f"rmse: {rmse}")
    
    small_val_mask = (torch.abs(golden_valid) < float(2e-14)) & (diff > float(2e-30))
    if small_val_mask.sum() > 0:
        small_val_error = torch.sum(diff[small_val_mask]).item()
        print(f"small val error: {small_val_error}")
    else:
        print(f"small val error: 0.0")
    
    return relative_max, relative_mean, rmse


def test_chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64():
    ptfile_path = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64_fp64/test_chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64_v2.pt'
    ptfile_path1 = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64_fp64/test_chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64_fp64_v2.pt'
    data = torch.load(ptfile_path, map_location=torch.device('cpu'), weights_only=False)
    datafp64 = torch.load(ptfile_path1, map_location=torch.device('cpu'), weights_only=False)
    
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    
    w = input_data['w']
    k = input_data['k']
    do = input_data['do']
    q = input_data['q']
    scale = input_data['scale']
    chunk_size = input_data['chunk_size']
    dv = input_data['dv']
    g = input_data['g']
    dh, dh0, dv2 = chunk_gated_delta_rule_bwd_dhu(q=q, k=k, w=w, do=do, dv=dv, g=g, scale=scale, )

    npu_out = {
        'dh': dh.cpu(),
        'dh0': dh0.cpu() if dh0 is not None else None,
        'dv2': dv2.cpu(),
    }
    
    # Compare with GPU fp64 output
    for key in datafp64['gpu_output']:
        if key in npu_out and key != "dh0":
            gpu_output = data['gpu_output'][key]
            gpu_output_fp64 = datafp64['gpu_output'][key]
            npu_output = npu_out[key]
            print(f"Comparing {key}:")
            # cal_precision(npu_output, gpu_output_fp64)
            relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(npu_output, gpu_output_fp64)
            relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(gpu_output, gpu_output_fp64)
            print(
                f"最大相对误差比 :{relative_max_npu / relative_max_gpu if relative_max_gpu != float('inf') else 'N/A (GPU inf)'}, "
                f"平均相对误差比 :{relative_mean_npu / relative_mean_gpu if relative_mean_gpu != float('inf') else 'N/A (GPU inf)'}, "
                f"均方根误差比 :{rmse_npu / rmse_gpu if rmse_gpu != float('inf') else 'N/A (GPU inf)'}")
            
            if relative_max_gpu != float('inf') and relative_mean_gpu != float('inf') and rmse_gpu != float('inf'):
                assert relative_max_npu / relative_max_gpu <= 2, "最大相对误差比 check failed"
                assert relative_mean_npu / relative_mean_gpu <= 1.2, "平均相对误差比 check failed"
                assert rmse_npu / rmse_gpu <= 1.2, "均方根误差比 check failed"
            else:
                print("WARNING: GPU reference has inf values, skipping ratio checks")
    print("Output comparison passed!")



if __name__ == '__main__':
    test_chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64()
