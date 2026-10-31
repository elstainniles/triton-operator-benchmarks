# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: fla/ops/generalized_delta_rule/iplr/fused_recurrent.py:122 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
# Source archive: dependencies and launch adaptation are not executable here.
# The complete original file and dependency snapshots are under upstream/.
import triton
import triton.language as tl


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'USE_DHT': lambda args: args['dht'] is not None,
    'USE_DH0': lambda args: args['dh0'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8, 16]
        for num_stages in [2, 3]
    ],
    key=["BK", "BV"],
    **autotune_cache_kwargs,
)
@triton.jit
def fused_recurrent_bwd_kernel(
    # B: batch_size, H: n_heads, T: seq_len, D: b_dhead
    # NV: number of split in the V dimension. NK: number of split in the K dimension
    q,  # query [B, H, L, K]
    k,  # key [B, H, L, V]
    v,  # value [B, H, L, V]
    a,  # a [B, H, L, K]
    b,  # b [B, H, L, K]
    ha,  # ha [B, H, L, V]
    dht,  # gradient of final state [B, H, K, V]
    dh0,  # gradient of initial state [B, H, K, V]
    do,  # gradient of output [B, H, L, V]
    dq,  # gradient of query [NV, B, H, L, K]
    dk,  # gradient of key [NV, B, H, L, K]
    dv,  # gradient of value [NK, B, H, L, V]
    da,  # gradient of a [NV, B, H, L, K]
    db,  # gradient of b [NV, B, H, L, K]
    dha,  # gradient of ha [NK, B, H, L, V]
    h0,  # initial state [B, H, K, V]
    scale,  # K ** -0.5
    cu_seqlens,  # cu_seqlens
    B,  # batch_size
    H,  # n_heads
    T,  # seq_len
    K: tl.constexpr,  # K
    V: tl.constexpr,  # V
    BK: tl.constexpr,  # BLOCK SIZE along the K dimension
    BV: tl.constexpr,  # BLOCK SIZE along the V dimension
    USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state h0
    USE_DH0: tl.constexpr,  # whether to use dh0
    USE_DHT: tl.constexpr,  # whether to use dht
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    dk += i_v * B * H * K * T
    db += i_v * B * H * K * T
    dq += i_v * B * H * K * T
    da += i_v * B * H * K * T
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
    mask_k = tl.arange(0, BK) < K
    mask_v = (tl.arange(0, BV) + i_v * BV) < V

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V + i_v * BV
    ha += (bos * H + i_h) * V + i_v * BV
    a += (bos * H + i_h) * K
    b += (bos * H + i_h) * K
    do += (bos * H + i_h) * V + i_v * BV
    dq += (bos * H + i_h) * K
    dk += (bos * H + i_h) * K
    dv += (bos * H + i_h) * V + i_v * BV
    da += (bos * H + i_h) * K
    db += (bos * H + i_h) * K
    dha += (bos * H + i_h) * V + i_v * BV

    p_q = q + tl.arange(0, BK) + (T - 1) * H*K
    p_k = k + tl.arange(0, BK) + (T - 1) * H*K
    p_v = v + tl.arange(0, BV) + (T - 1) * H*V
    p_ha = ha + tl.arange(0, BV) + (T - 1) * H*V
    p_a = a + tl.arange(0, BK) + (T - 1) * H*K
    p_b = b + tl.arange(0, BK) + (T - 1) * H*K
    p_do = do + tl.arange(0, BV) + (T - 1) * H*V
    p_dk = dk + tl.arange(0, BK) + (T - 1) * H*K
    p_dv = dv + tl.arange(0, BV) + (T - 1) * H*V
    p_dha = dha + tl.arange(0, BV) + (T - 1) * H*V
    p_db = db + tl.arange(0, BK) + (T - 1) * H*K
    p_da = da + tl.arange(0, BK) + (T - 1) * H*K
    p_dq = dq + tl.arange(0, BK) + (T - 1) * H*K

    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_DHT:
        p_ht = dht + i_nh * K * V + (tl.arange(0, BK)[:, None]) * V + ((i_v * BV + tl.arange(0, BV))[None, :])
        b_dh += tl.load(p_ht, mask=mask_k[:, None] & mask_v[None, :], other=0).to(tl.float32)

    for _ in range(T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32) * scale
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask_v, other=0).to(tl.float32)
        b_b = tl.load(p_b, mask=mask_k, other=0).to(tl.float32)
        b_a = tl.load(p_a, mask=mask_k, other=0).to(tl.float32)
        b_ha = tl.load(p_ha, mask=mask_v, other=0).to(tl.float32)

        b_dh += b_q[:, None] * b_do[None, :]
        d_k = tl.sum(b_dh * b_v[None, :], axis=1)
        d_v = tl.sum(b_dh * b_k[:, None], axis=0)
        tl.store(p_dk, d_k.to(p_dk.dtype.element_ty), mask=mask_k)
        tl.store(p_dv, d_v.to(p_dv.dtype.element_ty), mask=mask_v)

        b_dha = tl.sum(b_dh * b_b[:, None], axis=0)
        tl.store(p_dha, b_dha.to(p_dha.dtype.element_ty), mask=mask_v)
        b_db = tl.sum(b_dh * b_ha[None, :], axis=1)
        tl.store(p_db, b_db.to(p_db.dtype.element_ty), mask=mask_k)

        b_dh += b_dha[None, :] * b_a[:, None]
        p_do -= H*V
        p_q -= H*K
        p_k -= H*K
        p_v -= H*V
        p_dk -= H*K
        p_dv -= H*V
        p_b -= H*K
        p_db -= H*K
        p_a -= H*K
        p_dha -= H*V
        p_ha -= H*V

    if USE_DH0:
        p_dh0 = dh0 + i_nh * K * V + (tl.arange(0, BK)[:, None]) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), mask=mask_k[:, None] & mask_v[None, :])

    tl.debug_barrier()

    b_h = tl.zeros([BK, BV], dtype=tl.float32)

    if USE_INITIAL_STATE:
        mask_kv = mask_k[:, None] & mask_v[None, :]
        p_h0 = h0 + i_nh * K * V + (tl.arange(0, BK)[:, None]) * V + ((i_v * BV + tl.arange(0, BV))[None, :])
        b_h += tl.load(p_h0, mask=mask_kv, other=0).to(tl.float32)

    p_k = k + tl.arange(0, BK)
    p_v = v + tl.arange(0, BV)
    p_ha = ha + tl.arange(0, BV)
    p_do = do + tl.arange(0, BV)
    p_dha = dha + tl.arange(0, BV)
    p_da = da + tl.arange(0, BK)
    p_dq = dq + tl.arange(0, BK)
    p_b = b + tl.arange(0, BK)

    for i in range(0, T):
        b_dha = tl.load(p_dha, mask=mask_v, other=0).to(tl.float32)
        d_a = tl.sum(b_dha[None, :] * b_h, axis=1)
        tl.store(p_da, d_a.to(p_da.dtype.element_ty), mask=mask_k)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask_v, other=0).to(tl.float32)
        b_b = tl.load(p_b, mask=mask_k, other=0).to(tl.float32)
        b_ha = tl.load(p_ha, mask=mask_v, other=0).to(tl.float32)
        b_h += b_k[:, None] * b_v[None, :] + b_b[:, None] * b_ha[None, :]
        _d_q = b_h * b_do[None, :]
        d_q = tl.sum(_d_q, axis=1) * scale
        tl.store(p_dq, d_q.to(p_dq.dtype.element_ty), mask=mask_k)

        p_k += H*K
        p_do += H*V
        p_v += H*V
        p_da += H*K
        p_dha += H*V
        p_ha += H*V
        p_dq += H*K
        p_b += H*K
