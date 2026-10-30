# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: fla/ops/rwkv6/fused_recurrent.py:302 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
# Source archive: dependencies and launch adaptation are not executable here.
# The complete original file and dependency snapshots are under upstream/.
import triton
import triton.language as tl


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['dh0'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
    ],
    key=['BK', 'BV'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_rwkv6_bwd_kernel_dkv(
    q,  # query [B, H, T, K]/[B, T, H, K]
    k,  # key [B, H, T, V]/[B, T, H, V]
    v,  # value [B, H, T, V]/[B, T, H, V]
    w,  # log gate [B, H, T]/[B, T, H]
    u,  # bonus [B, H, K]
    do,  # gradient of output [B, H, T, V]/[B, T, H, V]
    dk,  # gradient of key [NV, B, H, T, K]/[NK, B, T, H, K]
    dk1,  # gradient of key_aux [NV, B, H, T, K]/[NK, B, T, H, K]
    dv,  # gradient of value [NK, B, H, T, V]/[NV, B, T, H, V]
    dh0,  # gradient of initial hidden state [N, H, K, V]
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
    IS_VARLEN: tl.constexpr,
):
    i_v, i_k, i_nh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    p_q = q + (bos + ((T-1) if not REVERSE else 0)) * H*K + i_h * K + o_k
    p_k = k + (bos + ((T-1) if not REVERSE else 0)) * H*K + i_h * K + o_k
    p_v = v + (bos + ((T-1) if not REVERSE else 0)) * H*V + i_h * V + o_v
    p_w = w + (bos + ((T-1) if not REVERSE else 0)) * H*K + i_h * K + o_k
    p_do = do + (bos + ((T-1) if not REVERSE else 0)) * H*V + i_h * V + o_v
    p_dk = dk + ((i_v * all + bos) + ((T-1) if not REVERSE else 0)) * H*K + i_h * K + o_k
    p_dk1 = dk1 + ((i_v * all + bos) + ((T-1) if not REVERSE else 0)) * H*K + i_h * K + o_k
    p_dv = dv + ((i_k * all + bos) + ((T-1) if not REVERSE else 0)) * H*V + i_h * V + o_v
    p_u = u + i_h * K + o_k

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_u = tl.load(p_u, mask=mask_k, other=0).to(tl.float32)

    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    for _ in range(T - 1, -1, -1):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32) * scale
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_w = tl.load(p_w, mask=mask_k, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask_v, other=0).to(tl.float32)
        b_dkv = b_q[:, None] * b_do[None, :]
        b_dk = tl.sum(b_dh * b_v[None, :], 1)
        tl.store(p_dk1, b_dk.to(p_dk1.dtype.element_ty), mask=mask_k)
        b_dk += tl.sum(b_dkv * b_u[:, None] * b_v[None, :], 1)
        b_dv = tl.sum((b_dh + (b_dkv * b_u[:, None])) * b_k[:, None], 0)

        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=mask_k)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), mask=mask_v)
        b_dh *= exp(b_w)[:, None]
        b_dh += b_dkv

        p_q += (-1 if not REVERSE else 1) * H*K
        p_k += (-1 if not REVERSE else 1) * H*K
        p_v += (-1 if not REVERSE else 1) * H*V
        p_w += (-1 if not REVERSE else 1) * H*K
        p_do += (-1 if not REVERSE else 1) * H*V
        p_dk += (-1 if not REVERSE else 1) * H*K
        p_dk1 += (-1 if not REVERSE else 1) * H*K
        p_dv += (-1 if not REVERSE else 1) * H*V

    if USE_INITIAL_STATE:
        p_dh0 = dh0 + i_nh * K*V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), mask=mask_h)
