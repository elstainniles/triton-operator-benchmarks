# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: tests/ops/ascend_ops/cv_operator/chunk_abc/chunk_abc.py:234 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
# Source archive: dependencies and launch adaptation are not executable here.
# The complete original file and dependency snapshots are under upstream/.
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['T'])
def chunk_abc_fwd_kernel_h(
    k,
    v,
    z,
    h,
    h0,
    ht,
    T,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
    NORMK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h = tl.make_block_ptr(h0 + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h += tl.load(p_h, boundary_check=(0, 1)).to(tl.float32)
    if NORMK:
        p_z0 = tl.make_block_ptr(z + i_bh * T*K, (T * K,), (1,), (i_k * BK,), (BK,), (0,))
    else:
        p_z0 = tl.make_block_ptr(z + i_bh * T*V, (T * V,), (1,), (i_v * BV,), (BV,), (0,))
    b_zp = tl.load(p_z0).to(tl.float32)
    for i_t in range(NT):
        p_k = tl.make_block_ptr(k + i_bh * T*K, (K, T), (1, K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_v = tl.make_block_ptr(v + i_bh * T*V, (T, V), (V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * NT*K*V + i_t * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))

        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        if NORMK:
            p_zc = tl.make_block_ptr(z + i_bh * T*K, (T * K,), (1,), ((i_t * BT + BT - 1) * K + i_k * BK,), (BK,), (0,))
            b_zc = tl.load(p_zc, boundary_check=(0,))
            b_r, b_zp = exp(b_zp - b_zc), b_zc
            b_h = b_h * b_r[:, None]
            b_k = exp(b_k - b_zc[:, None]).to(b_k.dtype)
        else:
            p_zc = tl.make_block_ptr(z + i_bh * T*V, (T * V,), (1,), ((i_t * BT + BT - 1) * V + i_v * BV,), (BV,), (0,))
            b_zc = tl.load(p_zc, boundary_check=(0,))
            b_r, b_zp = exp(b_zp - b_zc), b_zc
            b_h = b_h * b_r[None, :]
            b_v = exp(b_v - b_zc[None, :]).to(b_v.dtype)
        b_h += tl.dot(b_k, b_v, allow_tf32=False)

    if STORE_FINAL_STATE:
        p_h = tl.make_block_ptr(ht + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))
