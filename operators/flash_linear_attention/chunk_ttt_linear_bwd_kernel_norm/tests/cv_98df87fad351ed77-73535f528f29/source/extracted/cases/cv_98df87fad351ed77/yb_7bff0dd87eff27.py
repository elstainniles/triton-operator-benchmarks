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
    'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None,
    'USE_FINAL_STATE_GRADIENT_B': lambda args: args['dhbt'] is not None,
    'USE_INITIAL_STATE': lambda args: args['dh0'] is not None,
    'USE_INITIAL_STATE_B': lambda args: args['dhb0'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [2, 4, 8, 16]
    ],
    key=['BT', 'BK', 'BV'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_ttt_linear_bwd_kernel_norm(
    q,
    k,
    v,
    v_new,
    x,
    y,
    r,
    w,
    b,
    eta,
    h,
    dht,
    dhbt,
    dh0,
    dhb0,
    do,
    dh,
    dhb,
    dv,
    dv_new,
    dk,
    dw,
    db,
    cu_seqlens,
    chunk_offsets,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    USE_FINAL_STATE_GRADIENT_B: tl.constexpr,
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
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    # [BV]
    b_dhb = tl.zeros([BV], dtype=tl.float32)
    if USE_FINAL_STATE_GRADIENT:
        p_dht = tl.make_block_ptr(dht + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_dh += tl.load(p_dht, boundary_check=(0, 1), padding_option="zero")
    if USE_FINAL_STATE_GRADIENT_B:
        p_dhbt = tl.make_block_ptr(dhbt + i_nh * V, (V,), (1,), (i_v * BV,), (BV,), (0,))
        b_dhb += tl.load(p_dhbt, boundary_check=(0,), padding_option="zero")

    # [BV]
    offs_v = tl.arange(0, BV)
    offs_t = tl.arange(0, BT)
    b_w = tl.load(w + i_h * V + offs_v, mask=offs_v < V, other=0.)
    b_b = tl.load(b + i_h * V + offs_v, mask=offs_v < V, other=0.)
    b_dw = tl.zeros([BV], dtype=b_w.dtype)
    b_db = tl.zeros([BV], dtype=b_b.dtype)
    p_dw = tl.make_block_ptr(dw + i_nh * V, (V,), (1,), (i_v * BV,), (BV,), (0,))
    p_db = tl.make_block_ptr(db + i_nh * V, (V,), (1,), (i_v * BV,), (BV,), (0,))

    for i_t in range(NT - 1, -1, -1):
        p_h = tl.make_block_ptr(h + ((boh+i_t) * H + i_h) * K*V, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        p_dh = tl.make_block_ptr(dh + ((boh+i_t) * H + i_h) * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        p_dhb = tl.make_block_ptr(dhb + ((boh+i_t) * H + i_h) * V, (V,), (1,), (i_v * BV,), (BV,), (0,))
        tl.store(p_dh, b_dh.to(p_dh.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dhb, b_dhb.to(p_dhb.dtype.element_ty), boundary_check=(0,))
        p_q = tl.make_block_ptr(q+(bos*H+i_h)*K, (K, T), (1, H*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_k = tl.make_block_ptr(k+(bos*H+i_h)*K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_v = tl.make_block_ptr(v+(bos*H+i_h)*V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_v_new = tl.make_block_ptr(v_new+(bos*H+i_h)*V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_x = tl.make_block_ptr(x+(bos*H+i_h)*V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_y = tl.make_block_ptr(y+(bos*H+i_h)*V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dv_new = tl.make_block_ptr(dv_new+(bos*H+i_h)*V, (T, V), (H*V, 1), (i_t*BT, i_v * BV), (BT, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv+(bos*H+i_h)*V, (T, V), (H*V, 1), (i_t*BT, i_v * BV), (BT, BV), (1, 0))
        p_dk = tl.make_block_ptr(dk+(bos*H+i_h)*K, (T, K), (H*K, 1), (i_t*BT, i_k * BK), (BT, BK), (1, 0))
        p_do = tl.make_block_ptr(do+(bos*H+i_h)*V, (T, V), (H*V, 1), (i_t*BT, i_v * BV), (BT, BV), (1, 0))
        p_r = tl.make_block_ptr(r+bos*H+i_h, (T, 1), (H, 1), (i_t*BT, 0), (BT, 1), (1, 0))
        p_eta_last = eta+bos*H+i_h + (T-1)*H if i_t == NT-1 else eta+bos*H+i_h + (i_t*BT+BT-1)*H
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        b_dv_new = tl.load(p_dv_new, boundary_check=(0, 1), padding_option="zero").to(b_k.dtype)
        b_eta_last = tl.load(p_eta_last)
        b_dv_new -= tl.dot(b_eta_last * b_k, b_dh.to(b_k.dtype))
        b_dv_new -= b_eta_last * b_dhb.to(b_k.dtype)[None, :]

        b_v_new = tl.load(p_v_new, boundary_check=(0, 1), padding_option="zero")
        b_x = tl.load(p_x, boundary_check=(0, 1), padding_option="zero").to(b_k.dtype)
        b_y = tl.load(p_y, boundary_check=(0, 1), padding_option="zero").to(b_k.dtype)
        b_rstd = tl.load(p_r, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        b_dy = b_rstd * (b_dv_new * V - tl.sum(b_dv_new, axis=1, keep_dims=True) -
                         b_x * tl.sum(b_dv_new * b_x, axis=1, keep_dims=True)) / V
        b_dx = -b_rstd * (b_dv_new * tl.sum(b_x * b_y, axis=1, keep_dims=True) +
                          b_y * tl.sum(b_dv_new * b_x, axis=1, keep_dims=True)) / V
        b_drstd = tl.sum(b_dv_new.to(b_rstd.dtype) * b_v_new.to(b_rstd.dtype) / b_rstd, axis=1, keep_dims=True)

        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        b_w = b_w.to(b_k.dtype)
        b_b = b_b.to(b_k.dtype)
        b_dv = -b_w * b_dy.to(b_k.dtype)
        b_dk = b_w * b_dy.to(b_k.dtype)
        b_dw += tl.sum(2 * b_w * b_x * b_dy.to(b_k.dtype) +
                       (b_b - b_v.to(b_k.dtype) + b_k) * b_dy.to(b_k.dtype), axis=0).to(b_dw.dtype)
        b_db += tl.sum(b_w * b_dy.to(b_k.dtype), axis=0).to(b_db.dtype)
        b_dx = b_dx.to(b_k.dtype) + b_w * b_w * b_dy.to(b_k.dtype)

        # d_rstd, dx --> dkh --> dk, dh
        b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
        b_h = tl.load(p_h, boundary_check=(0, 1), padding_option="zero")
        b_do = tl.load(p_do, boundary_check=(0, 1), padding_option="zero")
        b_q = (b_q * scale).to(b_q.dtype)
        b_dkh = b_rstd * (V * b_dx - tl.sum(b_dx, axis=1, keep_dims=True) -
                          b_x * tl.sum(b_x * b_dx, axis=1, keep_dims=True)) / V
        b_dkh -= b_rstd * b_rstd * b_drstd * b_x / V
        b_dkh = tl.where((offs_v < V)[None, :] * (offs_t < T-i_t*BT)[:, None], b_dkh, 0.)
        b_dk += tl.dot(b_dkh, b_h.to(b_dkh.dtype)).to(b_k.dtype)
        b_dh += tl.dot(b_q, b_do.to(b_q.dtype)) + tl.dot(tl.trans(b_k).to(b_dkh.dtype), b_dkh)
        b_dhb += tl.sum(b_do + b_dkh, axis=0)
        b_dh = tl.where((offs_v < V)[None, :], b_dh, 0.)
        b_dhb = tl.where((offs_v < V), b_dhb, 0.)

        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dw, b_dw.to(p_dw.dtype.element_ty), boundary_check=(0,))
    tl.store(p_db, b_db.to(p_db.dtype.element_ty), boundary_check=(0,))

    if USE_INITIAL_STATE:
        p_dh0 = tl.make_block_ptr(dh0 + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), boundary_check=(0, 1))
    if USE_INITIAL_STATE_B:
        p_dhb0 = tl.make_block_ptr(dhb0+i_nh*V, (V,), (1,), (i_v * BV,), (BV,), (0,))
        tl.store(p_dhb0, b_dhb.to(p_dhb0.dtype.element_ty), boundary_check=(0,))

def chunk_ttt_linear_bwd_norm(
    q: torch.Tensor,  # [B, H, L, D]
    k: torch.Tensor,  # [B, H, L, D]
    v: torch.Tensor,  # [B, H, L, D]
    v_new: torch.Tensor,  # [B, H, L, D]
    x: torch.Tensor,  # [B, H, L, D]
    y: torch.Tensor,  # [B, H, L, D]
    rstd: torch.Tensor,  # [B, H, L, 1]
    w: torch.Tensor,  # [H, D]
    b: torch.Tensor,  # [H, D]
    eta: torch.Tensor,  # [B, H, L, 1]
    h0: torch.Tensor,  # [B, H, D, D]
    hb0: torch.Tensor,  # [B, H, 1, D]
    h: torch.Tensor,  # [B, H, NT, D, D]
    dht: torch.Tensor | None,  # [B, H, D, D]
    dhbt: torch.Tensor | None,  # [B, H, 1, D]
    dv_new: torch.Tensor | None,  # [B, H, L, D]
    do: torch.Tensor,  # [B, H, L, D]
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 16,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # torch implementation of `dkh, dw, db, dk, dv` for LN^2
    assert cu_seqlens is None, "bwd of varlen is not implemented yet."
    B, T, H, K, V = *q.shape, do.shape[-1]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT)

    BK = max(triton.next_power_of_2(K), 16)
    BV = max(triton.next_power_of_2(V), 16)
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    assert NK == 1, 'NK > 1 is not supported by TTT.'
    assert NV == 1, 'NV > 1 is not supported by TTT.'

    dh = q.new_empty(B, NT, H, K, V)
    dhb = q.new_empty(B, NT, H, 1, V)
    dh0 = torch.empty_like(h0, dtype=torch.float32) if h0 is not None else None
    dhb0 = torch.empty_like(hb0, dtype=torch.float32) if hb0 is not None else None
    dv = torch.empty_like(v)
    dk = torch.empty_like(k)
    dw = w.new_empty(B, H, V)
    db = b.new_empty(B, H, V)

    grid = (NK, NV, N * H)
    chunk_ttt_linear_bwd_kernel_norm[grid](
        q=q,
        k=k,
        v=v,
        v_new=v_new,
        x=x,
        y=y,
        r=rstd,
        w=w,
        b=b,
        eta=eta,
        h=h,
        dht=dht,
        dhbt=dhbt,
        dh0=dh0,
        dhb0=dhb0,
        do=do,
        dh=dh,
        dhb=dhb,
        dv=dv,
        dv_new=dv_new,
        dk=dk,
        dw=dw,
        db=db,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
    )
    dw = dw.sum(dim=0)
    db = db.sum(dim=0)
    return dh, dhb, dh0, dhb0, dv, dk, dw, db
