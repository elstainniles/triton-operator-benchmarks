# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch

import triton

import triton.language as tl

from yb_1858bf26d5878c import prepare_chunk_indices
from yb_1858bf26d5878c import prepare_chunk_offsets

from yb_74f55558e08fc1 import autocast_custom_bwd
from yb_74f55558e08fc1 import autocast_custom_fwd
from yb_74f55558e08fc1 import autotune_cache_kwargs
from yb_74f55558e08fc1 import input_guard

@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'USE_INITIAL_STATE_B': lambda args: args['hb0'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8]
    ],
    key=['BT', 'BK', 'BV'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_ttt_linear_bwd_kernel_h(
    k,
    v,
    v_new,
    eta,
    w,
    b,
    eps,
    h,
    h0,
    hb0,
    x,
    y,
    r,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_INITIAL_STATE_B: tl.constexpr,
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

    # [BK, BV]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    # [BV]
    b_hb = tl.zeros([BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = tl.make_block_ptr(h0 + i_nh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h = tl.load(p_h0, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    if USE_INITIAL_STATE_B:
        p_hb0 = tl.make_block_ptr(hb0 + i_nh * V, (V,), (1,), (i_v * BV,), (BV,), (0,))
        b_hb = tl.load(p_hb0, boundary_check=(0,), padding_option="zero").to(tl.float32)

    offs = tl.arange(0, BV)
    b_w = tl.load(w + i_h * V + offs, mask=offs < V, other=0.)
    b_b = tl.load(b + i_h * V + offs, mask=offs < V, other=0.)

    for i_t in range(NT):
        p_h = tl.make_block_ptr(h + ((boh + i_t) * H + i_h) * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))
        p_k = tl.make_block_ptr(k+(bos*H+i_h)*K, (K, T), (1, H*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_v = tl.make_block_ptr(v+(bos*H+i_h)*V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_v_new = tl.make_block_ptr(v_new+(bos*H+i_h)*V, (T, V), (H*V, 1), (i_t*BT, i_v * BV), (BT, BV), (1, 0))
        p_x = tl.make_block_ptr(x+(bos*H+i_h)*V, (T, V), (H*V, 1), (i_t*BT, i_v * BV), (BT, BV), (1, 0))
        p_y = tl.make_block_ptr(y+(bos*H+i_h)*V, (T, V), (H*V, 1), (i_t*BT, i_v * BV), (BT, BV), (1, 0))
        p_r = tl.make_block_ptr(r+bos*H+i_h, (T, 1), (H, 1), (i_t*BT, 0), (BT, 1), (1, 0))
        p_eta_last = eta+bos*H+i_h + (T-1)*H if i_t == NT-1 else eta+bos*H+i_h + (i_t*BT+BT-1)*H
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")

        b_kh = tl.dot(tl.trans(b_k), b_h.to(b_k.dtype), allow_tf32=False).to(tl.float32) + b_hb[None, :]
        b_kh = tl.where((offs < V)[None, :], b_kh, 0.)
        mean = tl.sum(b_kh, axis=1, keep_dims=True) / V
        xbar = tl.where((offs < V)[None, :], b_kh - mean, 0.)
        var = tl.sum(xbar * xbar, axis=1, keep_dims=True) / V
        rstd = 1 / tl.sqrt(var.to(tl.float32) + eps)
        b_kh_hat = (b_kh - mean) * rstd

        b_v = b_kh_hat.to(b_k.dtype) * b_w[None, :].to(b_k.dtype) + \
            b_b[None, :].to(b_k.dtype) - b_v.to(b_k.dtype) + tl.trans(b_k)
        b_v = tl.where((offs < V)[None, :], b_v * b_w[None, :].to(b_k.dtype), 0.)
        b_v2 = rstd * (V * b_v - tl.sum(b_v, axis=1, keep_dims=True) - b_kh_hat.to(b_k.dtype)
                       * tl.sum(b_v * b_kh_hat.to(b_k.dtype), axis=1, keep_dims=True)) / V
        tl.store(p_x, b_kh_hat.to(p_x.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_y, b_v.to(p_y.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_r, rstd.to(p_r.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_v_new, b_v2.to(p_v_new.dtype.element_ty), boundary_check=(0, 1))
        b_eta_last = tl.load(p_eta_last)
        b_h = b_h - tl.dot(b_eta_last * b_k, b_v2.to(b_k.dtype), allow_tf32=False)
        b_hb = b_hb - tl.sum(b_eta_last * b_v2.to(b_k.dtype), axis=0)

def chunk_ttt_linear_bwd_h(
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    eta: torch.Tensor,
    eps: float,
    initial_state: torch.Tensor | None = None,
    initial_state_bias: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 16,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT)
    BK = max(triton.next_power_of_2(K), 16)
    BV = max(triton.next_power_of_2(V), 16)
    assert max(BK, BV) <= 128, "current kernel does not support head dimension larger than 128."
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    assert NK == 1, 'NK > 1 is not supported because it involves time-consuming synchronization'
    assert NV == 1, 'NV > 1 is not supported by TTT update rule.'

    h = k.new_empty(B, NT, H, K, V)
    rstd = v.new_empty(B, T, H, 1, dtype=torch.float32)
    x = torch.empty_like(v)
    y = torch.empty_like(v)

    v_new = torch.empty_like(v)
    grid = (NK, NV, N * H)

    chunk_ttt_linear_bwd_kernel_h[grid](
        k=k,
        v=v,
        v_new=v_new,
        eta=eta,
        w=w,
        b=b,
        eps=eps,
        h=h,
        h0=initial_state,
        hb0=initial_state_bias,
        x=x,
        y=y,
        r=rstd,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        NT=NT,
    )
    return h, v_new, x, y, rstd
