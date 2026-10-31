# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: fla/ops/rwkv7/fused_recurrent.py:34 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
# Source archive: dependencies and launch adaptation are not executable here.
# The complete original file and dependency snapshots are under upstream/.
import triton
import triton.language as tl


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BV in [16, 32, 64]
        for num_warps in [2, 4, 8, 16]
        for num_stages in [2, 3, 4]
    ],
    key=['BK'],
    use_cuda_graph=USE_CUDA_GRAPH,
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_rwkv7_fwd_kernel(
    r,
    w,
    k,
    v,
    kk,
    a,
    o,
    h0,
    ht,
    cu_seqlens,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    REVERSE: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_DECODE: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H

    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    p_r = r + (bos + ((T - 1) if REVERSE else 0)) * H*K + i_h * K + o_k
    p_w = w + (bos + ((T - 1) if REVERSE else 0)) * H*K + i_h * K + o_k
    p_k = k + (bos + ((T - 1) if REVERSE else 0)) * H*K + i_h * K + o_k
    p_v = v + (bos + ((T - 1) if REVERSE else 0)) * H*V + i_h * V + o_v
    p_a = a + (bos + ((T - 1) if REVERSE else 0)) * H*K + i_h * K + o_k
    p_kk = kk + (bos + ((T - 1) if REVERSE else 0)) * H*K + i_h * K + o_k

    p_o = o + (bos + ((T - 1) if REVERSE else 0)) * H*V + i_h * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh * K*V + o_k[:, None] * V + o_v
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    if IS_DECODE:
        b_r = tl.load(p_r, mask=mask_k, other=0).to(tl.float32) * scale
        b_w = tl.load(p_w, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_a = tl.load(p_a, mask=mask_k, other=0).to(tl.float32)
        b_kk = tl.load(p_kk, mask=mask_k, other=0).to(tl.float32)
        b_act_a = -b_kk
        b_b = b_kk * b_a

        b_h = exp(b_w)[:, None] * b_h + b_b[:, None] * tl.sum(b_act_a[:, None] * b_h, 0)[None, :]
        b_h += b_k[:, None] * b_v[None, :]
        b_o = tl.sum(b_h * b_r[:, None], 0)

        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
    else:
        for _ in range(0, T):
            b_r = tl.load(p_r, mask=mask_k, other=0).to(tl.float32) * scale
            b_w = tl.load(p_w, mask=mask_k, other=0).to(tl.float32)
            b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
            b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
            b_a = tl.load(p_a, mask=mask_k, other=0).to(tl.float32)
            b_kk = tl.load(p_kk, mask=mask_k, other=0).to(tl.float32)
            b_act_a = -b_kk
            b_b = b_kk * b_a

            b_h = exp(b_w)[:, None] * b_h + b_b[:, None] * tl.sum(b_act_a[:, None] * b_h, 0)[None, :]
            b_h += b_k[:, None] * b_v[None, :]
            b_o = tl.sum(b_h * b_r[:, None], 0)

            tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
            p_r += (-1 if REVERSE else 1) * H*K
            p_w += (-1 if REVERSE else 1) * H*K
            p_k += (-1 if REVERSE else 1) * H*K
            p_v += (-1 if REVERSE else 1) * H*V
            p_a += (-1 if REVERSE else 1) * H*K
            p_kk += (-1 if REVERSE else 1) * H*K
            p_o += (-1 if REVERSE else 1) * H*V

    if STORE_FINAL_STATE:
        p_ht = ht + i_nh * K*V + o_k[:, None] * V + o_v
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)
