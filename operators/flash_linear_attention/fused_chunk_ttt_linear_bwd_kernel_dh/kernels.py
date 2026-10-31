# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: fla/ops/ttt/fused_chunk.py:277 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
# Source archive: dependencies and launch adaptation are not executable here.
# The complete original file and dependency snapshots are under upstream/.
import triton
import triton.language as tl


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['dh0'] is not None,
    'USE_INITIAL_STATE_B': lambda args: args['dhb0'] is not None,
    'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None,
    'USE_FINAL_STATE_GRADIENT_B': lambda args: args['dhbt'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in NUM_WARPS
    ],
    key=['BT', 'BK', 'BV'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def fused_chunk_ttt_linear_bwd_kernel_dh(
    q,
    k,
    v,
    v2,
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
    dk,
    dv,
    de,
    dw,
    db,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_INITIAL_STATE_B: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    USE_FINAL_STATE_GRADIENT_B: tl.constexpr,
):
    i_nh = tl.program_id(0)
    i_n, i_h = i_nh // H, i_nh % H
    bos, _ = i_n * T, i_n * T + T
    NT = tl.cdiv(T, BT)
    boh = i_n * NT

    # [BK, BV]
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    # [BV]
    b_dhb = tl.zeros([BV], dtype=tl.float32)
    if USE_FINAL_STATE_GRADIENT:
        p_dht = tl.make_block_ptr(dht + i_nh * K*V, (K, V), (V, 1), (0, 0), (BK, BV), (1, 0))
        b_dh += tl.load(p_dht, boundary_check=(0, 1), padding_option="zero")
    if USE_FINAL_STATE_GRADIENT_B:
        p_dhbt = tl.make_block_ptr(dhbt + i_nh * V, (V,), (1,), (0,), (BV,), (0,))
        b_dhb += tl.load(p_dhbt, boundary_check=(0,), padding_option="zero")

    # [BV]
    o_i = tl.arange(0, BT)
    v_i = tl.arange(0, BV)
    m_A = o_i[:, None] >= o_i[None, :]
    m_A_t = o_i[:, None] <= o_i[None, :]
    b_w = tl.load(w + i_h * V + v_i, mask=v_i < V, other=0.)
    b_b = tl.load(b + i_h * V + v_i, mask=v_i < V, other=0.)
    b_dw = tl.zeros([BV], dtype=b_w.dtype)
    b_db = tl.zeros([BV], dtype=b_b.dtype)
    p_dw = tl.make_block_ptr(dw + i_nh * V, (V,), (1,), (0,), (BV,), (0,))
    p_db = tl.make_block_ptr(db + i_nh * V, (V,), (1,), (0,), (BV,), (0,))

    for i_t in range(NT - 1, -1, -1):
        p_h = tl.make_block_ptr(h+((boh+i_t)*H+i_h)*K*V, (V, K), (1, V), (0, 0), (BV, BK), (0, 1))
        p_q = tl.make_block_ptr(q+(bos*H+i_h)*K, (K, T), (1, H*K), (0, i_t*BT), (BK, BT), (0, 1))
        p_k = tl.make_block_ptr(k+(bos*H+i_h)*K, (T, K), (H*K, 1), (i_t*BT, 0), (BT, BK), (1, 0))
        p_v = tl.make_block_ptr(v+(bos*H+i_h)*V, (T, V), (H*V, 1), (i_t*BT, 0), (BT, BV), (1, 0))
        p_v2 = tl.make_block_ptr(v2+(bos*H+i_h)*V, (T, V), (H*V, 1), (i_t*BT, 0), (BT, BV), (1, 0))
        p_x = tl.make_block_ptr(x+(bos*H+i_h)*V, (T, V), (H*V, 1), (i_t*BT, 0), (BT, BV), (1, 0))
        p_y = tl.make_block_ptr(y+(bos*H+i_h)*V, (T, V), (H*V, 1), (i_t*BT, 0), (BT, BV), (1, 0))
        p_r = tl.make_block_ptr(r+bos*H+i_h, (T, 1), (H, 1), (i_t*BT, 0), (BT, 1), (1, 0))
        p_e = tl.make_block_ptr(eta+(bos*H+i_h), (T,), (H,), (i_t*BT,), (BT,), (0,))
        p_dv = tl.make_block_ptr(dv+(bos*H+i_h)*V, (T, V), (H*V, 1), (i_t*BT, 0), (BT, BV), (1, 0))
        p_dk = tl.make_block_ptr(dk+(bos*H+i_h)*K, (T, K), (H*K, 1), (i_t*BT, 0), (BT, BK), (1, 0))
        p_do = tl.make_block_ptr(do+(bos*H+i_h)*V, (T, V), (H*V, 1), (i_t*BT, 0), (BT, BV), (1, 0))
        p_de = tl.make_block_ptr(de+(bos*H+i_h), (T,), (H,), (i_t*BT,), (BT,), (0,))
        p_e_last = eta+bos*H+i_h + (T-1)*H if i_t == NT-1 else eta+bos*H+i_h + (i_t*BT+BT-1)*H
        b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        b_e = tl.load(p_e, boundary_check=(0,), padding_option="zero")
        b_do = tl.load(p_do, boundary_check=(0, 1), padding_option="zero")
        b_e_last = tl.load(p_e_last)
        b_A = tl.dot(b_k, b_q)
        b_A = - tl.where(m_A_t, b_A * scale * b_e[None, :], 0).to(do.dtype.element_ty)
        b_Ae = - tl.where(m_A_t, b_e[None, :], 0).to(do.dtype.element_ty)
        b_dv_new = tl.dot(b_A.to(b_do.dtype), b_do) + tl.dot(b_Ae.to(b_do.dtype), b_do)
        b_dv_new -= tl.dot(b_e_last * b_k, b_dh.to(b_k.dtype))
        b_dv_new -= b_e_last * b_dhb.to(b_k.dtype)[None, :]

        b_v2 = tl.load(p_v2, boundary_check=(0, 1), padding_option="zero").to(b_k.dtype)
        b_x = tl.load(p_x, boundary_check=(0, 1), padding_option="zero").to(b_k.dtype)
        b_y = tl.load(p_y, boundary_check=(0, 1), padding_option="zero").to(b_k.dtype)
        b_rstd = tl.load(p_r, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        b_dy = b_rstd * (b_dv_new * V - tl.sum(b_dv_new, axis=1, keep_dims=True) -
                         b_x * tl.sum(b_dv_new * b_x, axis=1, keep_dims=True)) / V
        b_dx = -b_rstd * (b_dv_new * tl.sum(b_x * b_y, axis=1, keep_dims=True) +
                          b_y * tl.sum(b_dv_new * b_x, axis=1, keep_dims=True)) / V
        b_drstd = tl.sum(b_dv_new.to(b_rstd.dtype) * b_v2.to(b_rstd.dtype) / b_rstd, axis=1, keep_dims=True)

        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        b_w = b_w.to(b_k.dtype)
        b_b = b_b.to(b_k.dtype)
        b_dv = -b_w * b_dy.to(b_k.dtype)
        b_dk = b_w * b_dy.to(b_k.dtype)
        b_dw += tl.sum(2 * b_w * b_x * b_dy.to(b_k.dtype) +
                       (b_b - b_v.to(b_k.dtype) + b_k) * b_dy.to(b_k.dtype), axis=0).to(b_dw.dtype)
        b_db += tl.sum(b_w * b_dy.to(b_k.dtype), axis=0).to(b_db.dtype)
        b_dx = b_dx.to(b_k.dtype) + b_w * b_w * b_dy.to(b_k.dtype)

        b_h = tl.load(p_h, boundary_check=(0, 1), padding_option="zero")
        b_q = (b_q * scale).to(b_q.dtype)
        b_dkh = b_rstd * (V * b_dx - tl.sum(b_dx, axis=1, keep_dims=True) -
                          b_x * tl.sum(b_x * b_dx, axis=1, keep_dims=True)) / V
        b_dkh -= b_rstd * b_rstd * b_drstd * b_x / V
        b_dkh = tl.where((v_i < V)[None, :] * (o_i < T-i_t*BT)[:, None], b_dkh, 0.)
        b_dk += tl.dot(b_dkh, b_h.to(b_dkh.dtype)).to(b_k.dtype)

        b_ds = tl.dot(b_do, tl.trans(b_v2))
        b_ds = tl.where(m_A, b_ds, 0)
        b_ds = b_ds.to(b_k.dtype)
        i_last = (BT-1) if (i_t*BT+BT) <= T else (T % BT-1)
        mask = (o_i == i_last)
        b_dk -= b_e_last * tl.dot(b_v2, tl.trans(b_dh).to(b_v2.dtype))
        b_dk -= tl.dot(tl.trans(b_ds), tl.trans(b_q) * b_e[:, None])
        b_de = mask * tl.sum(- b_dh * tl.trans(tl.dot(tl.trans(b_v2), b_k))).to(b_k.dtype)
        b_de -= mask * tl.sum(b_dhb * tl.sum(b_v2, axis=0)).to(b_k.dtype)
        b_de -= tl.sum(tl.dot(b_ds, b_k) * tl.trans(b_q).to(b_k.dtype), axis=1)
        b_de -= tl.sum(b_ds, axis=1)
        b_dh += tl.dot(b_q, b_do.to(b_q.dtype)) + tl.dot(tl.trans(b_k).to(b_dkh.dtype), b_dkh)
        b_dhb += tl.sum(b_do + b_dkh, axis=0)
        b_dh = tl.where((v_i < V)[None, :], b_dh, 0.)
        b_dhb = tl.where((v_i < V), b_dhb, 0.)

        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_de, b_de.to(p_de.dtype.element_ty), boundary_check=(0,))
    tl.store(p_dw, b_dw.to(p_dw.dtype.element_ty), boundary_check=(0,))
    tl.store(p_db, b_db.to(p_db.dtype.element_ty), boundary_check=(0,))

    if USE_INITIAL_STATE:
        p_dh0 = tl.make_block_ptr(dh0+i_nh*K*V, (K, V), (V, 1), (0, 0), (BK, BV), (1, 0))
        tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), boundary_check=(0, 1))
    if USE_INITIAL_STATE_B:
        p_dhb0 = tl.make_block_ptr(dhb0+i_nh*V, (V,), (1,), (0,), (BV,), (0,))
        tl.store(p_dhb0, b_dhb.to(p_dhb0.dtype.element_ty), boundary_check=(0,))
