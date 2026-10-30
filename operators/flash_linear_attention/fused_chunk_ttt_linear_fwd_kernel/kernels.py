# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: fla/ops/ttt/fused_chunk.py:34 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
# Source archive: dependencies and launch adaptation are not executable here.
# The complete original file and dependency snapshots are under upstream/.
import triton
import triton.language as tl


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'USE_INITIAL_STATE_B': lambda args: args['hb0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
    ],
    key=['BT', 'BK', 'BV'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def fused_chunk_ttt_linear_fwd_kernel(
    q,
    k,
    v,
    eta,
    w,
    b,
    o,
    scale,
    eps,
    h0,
    hb0,
    ht,
    hbt,
    cu_seqlens,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_INITIAL_STATE_B: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_nh = tl.program_id(0)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)

    o_i = tl.arange(0, BT)
    v_i = tl.arange(0, BV)
    m_A = o_i[:, None] >= o_i[None, :]
    b_w = tl.load(w + i_h * V + v_i, mask=v_i < V, other=0.)
    b_b = tl.load(b + i_h * V + v_i, mask=v_i < V, other=0.)

    # [BK, BV]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    # [BV]
    b_hb = tl.zeros([BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = tl.make_block_ptr(h0 + i_nh * K * V, (K, V), (V, 1), (0, 0), (BK, BV), (1, 0))
        b_h = tl.load(p_h0, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    if USE_INITIAL_STATE_B:
        p_hb0 = tl.make_block_ptr(hb0 + i_nh * V, (V,), (1,), (0,), (BV,), (0,))
        b_hb = tl.load(p_hb0, boundary_check=(0,), padding_option="zero").to(tl.float32)

    for i_t in range(NT):
        p_q = tl.make_block_ptr(q+(bos*H+i_h)*K, (T, K), (H*K, 1), (i_t*BT, 0), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k+(bos*H+i_h)*K, (K, T), (1, H*K), (0, i_t*BT), (BK, BT), (0, 1))
        p_v = tl.make_block_ptr(v+(bos*H+i_h)*V, (T, V), (H*V, 1), (i_t*BT, 0), (BT, BV), (1, 0))
        p_o = tl.make_block_ptr(o+(bos*H+i_h)*V, (T, V), (H*V, 1), (i_t*BT, 0), (BT, BV), (1, 0))
        p_e = tl.make_block_ptr(eta+(bos*H+i_h), (T,), (H,), (i_t*BT,), (BT,), (0,))
        p_e_last = eta+bos*H+i_h + (T-1)*H if i_t == NT-1 else eta+bos*H+i_h + (i_t*BT+BT-1)*H
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")

        # [BT, BV]
        b_kh = tl.dot(tl.trans(b_k), b_h.to(b_k.dtype), allow_tf32=False).to(tl.float32) + b_hb[None, :]
        b_kh = tl.where((v_i < V)[None, :], b_kh, 0.)
        mean = tl.sum(b_kh, axis=1, keep_dims=True) / V
        xbar = tl.where((v_i < V)[None, :], b_kh - mean, 0.)
        var = tl.sum(xbar * xbar, axis=1, keep_dims=True) / V
        rstd = 1 / tl.sqrt(var.to(tl.float32) + eps)
        b_kh_hat = (b_kh - mean) * rstd

        b_v = b_kh_hat.to(b_k.dtype) * b_w[None, :].to(b_k.dtype) + \
            b_b[None, :].to(b_k.dtype) - b_v.to(b_k.dtype) + tl.trans(b_k)
        b_v = tl.where((v_i < V)[None, :], b_v * b_w[None, :].to(b_k.dtype), 0.)
        b_v2 = rstd * (V * b_v - tl.sum(b_v, axis=1, keep_dims=True) - b_kh_hat.to(b_k.dtype)
                       * tl.sum(b_v * b_kh_hat.to(b_k.dtype), axis=1, keep_dims=True)) / V

        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
        # [BT]
        b_e = tl.load(p_e, boundary_check=(0,), padding_option="zero")
        b_q = (b_q * scale).to(b_k.dtype)

        # [BT, BT]
        b_A = tl.dot(b_q, b_k, allow_tf32=False)
        b_A = tl.where(m_A, b_A, 0)
        b_Ae = tl.where(m_A, b_e[:, None], 0.0)

        b_o = - tl.dot(b_e[:, None] * b_A.to(b_v2.dtype), b_v2, allow_tf32=False)
        b_o += b_hb[None, :] - tl.dot(b_Ae.to(b_v2.dtype), b_v2, allow_tf32=False)
        b_o += tl.dot(b_q, b_h.to(b_q.dtype), allow_tf32=False)
        b_e_last = tl.load(p_e_last)
        b_h = b_h - tl.dot(b_e_last * b_k, b_v2.to(b_k.dtype), allow_tf32=False)
        b_hb = b_hb - tl.sum(b_e_last * b_v2.to(b_k.dtype), axis=0)
        b_h = tl.where((v_i < V)[None, :], b_h, 0.)
        b_hb = tl.where((v_i < V), b_hb, 0.)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht + i_nh * K*V, (K, V), (V, 1), (0, 0), (BK, BV), (1, 0))
        p_hbt = tl.make_block_ptr(hbt + i_nh * V, (V,), (1,), (0,), (BV,), (0,))
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_hbt, b_hb.to(p_hbt.dtype.element_ty), boundary_check=(0,))
