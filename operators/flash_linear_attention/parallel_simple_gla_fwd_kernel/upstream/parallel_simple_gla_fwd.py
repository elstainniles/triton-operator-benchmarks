import torch
import triton
import triton.language as tl

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from utils import exp

FLA_CACHE_RESULTS = os.getenv('FLA_CACHE_RESULTS', '1') == '1'
SUPPORTS_AUTOTUNE_CACHE = "cache_results" in triton.autotune.__code__.co_varnames
autotune_cache_kwargs = {"cache_results": FLA_CACHE_RESULTS} if SUPPORTS_AUTOTUNE_CACHE else {}

@triton.jit
def exp(x):
    return tl.exp(x.to(tl.float32))

# @triton.autotune(
#     configs=[
#         triton.Config({}, num_warps=num_warps, num_stages=num_stages)
#         for num_warps in [2, 4, 8, 16]
#         for num_stages in [2, 3, 4]
#     ],
#     key=["BT", "BS", "BK", "BV", "USE_G"],
#     **autotune_cache_kwargs,
# )
# @triton.heuristics({
#     'NV': lambda args: triton.cdiv(args['V'], args['BV']),
#     'OUTPUT_ATTENTIONS': lambda args: args['attn'] is not None,
#     'USE_G': lambda args: args['g'] is not None,
#     'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
# })
@triton.jit
def parallel_simple_gla_fwd_kernel(
    q,
    k,
    v,
    g,
    o,
    attn,
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
    OUTPUT_ATTENTIONS: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_G: tl.constexpr,
):
    i_kv, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_k, i_v = i_kv // NV, i_kv % NV
    i_b, i_h = i_bh // H, i_bh % H

    all = B * T
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    o += ((i_k * all + bos) * H + i_h) * V
    if USE_G:
        g += bos * H + i_h
    if OUTPUT_ATTENTIONS:
        attn += i_k * B * H * T * T + (bos * H + i_h * T) * T

    p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))

    # the Q block is kept in the shared memory throughout the whole kernel
    # [BT, BK]
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)
    b_o = tl.zeros([BT, BV], dtype=tl.float32)

    # [BT]
    o_q = i_t * BT + tl.arange(0, BT)
    m_q = o_q < T
    # Q block and K block have overlap.
    # masks required
    if USE_G:
        # [BT,]
        b_gq = tl.load(g + o_q * H, mask=m_q, other=float('-inf')).to(tl.float32)
        # rescale interchunk output
    else:
        b_gq = None

    for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
        p_k = tl.make_block_ptr(k, (K, T), (1, H*K), (i_k * BK, i_s), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v, (T, V), (H*V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))

        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        # [BK, BS]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BS, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BT, BS]
        m_s = (o_q[:, None] >= o_k[None, :]) & (m_q[:, None] & m_k[None, :])
        b_s = tl.dot(b_q, b_k)
        if USE_G:
            b_gk = tl.load(g + o_k * H, mask=m_k, other=0)
            b_s *= exp(b_gq[:, None] - b_gk[None, :])
        b_s = tl.where(m_s, b_s, 0)
        # [BT, BV]
        if i_s >= 0:
            b_o += tl.dot(b_s.to(b_q.dtype), b_v)
        if OUTPUT_ATTENTIONS:
            p_a = tl.make_block_ptr(attn, (T, T), (T, 1), (i_t * BT, i_s), (BT, BS), (1, 0))
            tl.store(p_a, b_s.to(p_a.dtype.element_ty), boundary_check=(0, 1))
    for i_s in range(i_t * BT - BS, -BS, -BS):
        p_k = tl.make_block_ptr(k, (K, T), (1, H*K), (i_k * BK, i_s), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v, (T, V), (H*V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))

        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        # [BK, BS]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BS, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BT, BS]
        m_s = m_q[:, None] & m_k[None, :]
        b_s = tl.dot(b_q, b_k)
        if USE_G:
            b_g = tl.load(g + o_k * H, mask=m_k, other=0)
            b_gn = tl.load(g + (min(i_s + BS, T) - 1) * H)
            b_gp = tl.load(g + (i_s-1) * H) if i_s % BT > 0 else 0.
            # No concrete meaning. Just to avoid some layout bugs.
            b_s *= exp(b_gq[:, None] + (b_gn - b_g)[None, :])
            b_gq += b_gn - b_gp
        b_s = tl.where(m_s, b_s, 0)
        if OUTPUT_ATTENTIONS:
            p_a = tl.make_block_ptr(attn, (T, T), (T, 1), (i_t * BT, i_s), (BT, BS), (1, 0))
            tl.store(p_a, b_s.to(p_a.dtype.element_ty), boundary_check=(0, 1))
        if i_s >= 0:
            b_o += tl.dot(b_s.to(b_v.dtype), b_v)
    p_o = tl.make_block_ptr(o, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def parallel_simple_gla_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: float,
    output_attentions: bool = False,
    chunk_size: int = 128,
):
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT, BS = chunk_size, 32
    BK = min(64, triton.next_power_of_2(K))
    BV = min(64, triton.next_power_of_2(V))

    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    assert BT % BS == 0

    NT = triton.cdiv(T, BT)
    grid = (NK * NV, NT, B * H)
    o = torch.empty(NK, *v.shape, dtype=v.dtype if NK == 1 else torch.float, device=q.device)
    attn = q.new_zeros(NK, B, H, T, T) if output_attentions else None

    parallel_simple_gla_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        o=o,
        attn=attn,
        scale=scale,
        cu_seqlens=None,
        chunk_indices=None,
        B=B,
        H=H,
        T=T,
        K=K,
        V=V,
        BT=BT,
        BS=BS,
        BK=BK,
        BV=BV,
        NV=NV,
        OUTPUT_ATTENTIONS=output_attentions,
        USE_G=False,
        IS_VARLEN=False,
    )
    o = o.sum(0)

    if output_attentions:
        attn = attn.sum(0)
    return o, g, attn


# @triton.heuristics({
#     'NV': lambda args: triton.cdiv(args['V'], args['BV']),
#     'OUTPUT_ATTENTIONS': lambda args: args['attn'] is not None,
#     'USE_G': lambda args: args['g'] is not None,
#     'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
# })
