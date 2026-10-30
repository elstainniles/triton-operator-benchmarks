# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: fla/ops/rwkv6/fused_recurrent.py:117 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
# Source archive: dependencies and launch adaptation are not executable here.
# The complete original file and dependency snapshots are under upstream/.
import triton
import triton.language as tl


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
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
def fused_recurrent_rwkv6_bwd_kernel_dq(
    k,  # key [B, H, T, V]/[B, T, H, V]
    v,  # value [B, H, T, V]/[B, T, H, V]
    w,  # log gate [B, H, T]/[B, T, H]
    u,  # bonus [B, H, K]
    do,  # gradient of output [B, H, T, V]/[B, T, H, V]
    dq,  # gradient of query [NV, B, H, T, K]/[NV, B, T, H, K]
    dq1,  # gradient of query_aux [NV, B, H, T, K]/[NV, B, T, H, K]
    h0,
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
    p_k = k + (bos + ((T-1) if REVERSE else 0)) * H*K + i_h * K + o_k
    p_v = v + (bos + ((T-1) if REVERSE else 0)) * H*V + i_h * V + o_v
    p_w = w + (bos + ((T-1) if REVERSE else 0)) * H*K + i_h * K + o_k
    p_do = do + (bos + ((T-1) if REVERSE else 0)) * H*V + i_h * V + o_v
    p_dq = dq + ((i_v * all + bos) + ((T-1) if REVERSE else 0)) * H*K + i_h * K + o_k
    p_dq1 = dq1 + ((i_v * all + bos) + ((T-1) if REVERSE else 0)) * H*K + i_h * K + o_k
    p_u = u + i_h * K + o_k

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_u = tl.load(p_u, mask=mask_k, other=0).to(tl.float32)

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh * K*V + o_k[:, None] * V + o_v[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for _ in range(0, T):
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_w = tl.load(p_w, mask=mask_k, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask_v, other=0).to(tl.float32)
        b_kv = b_k[:, None] * b_v[None, :]

        b_hq = b_h * b_do[None, :]
        b_dq = tl.sum(b_hq + b_kv * b_u[:, None] * b_do[None, :], 1) * scale
        b_dq1 = tl.sum(b_hq, 1)
        b_h = b_h * exp(b_w)[:, None]
        b_h += b_kv
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), mask=mask_k)
        tl.store(p_dq1, b_dq1.to(p_dq1.dtype.element_ty), mask=mask_k)

        p_k += (-1 if REVERSE else 1) * H*K
        p_v += (-1 if REVERSE else 1) * H*V
        p_w += (-1 if REVERSE else 1) * H*K
        p_do += (-1 if REVERSE else 1) * H*V
        p_dq += (-1 if REVERSE else 1) * H*K
        p_dq1 += (-1 if REVERSE else 1) * H*K
