# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: tests/ops/ascend_ops/cv_operator/chunkwise_bwd_kernel_diag/test_chunkwise_bwd_kernel_diag_npu.py:67 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
# Source archive: dependencies and launch adaptation are not executable here.
# The complete original file and dependency snapshots are under upstream/.
import triton
import triton.language as tl


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [4]
        for num_stages in [2, 3, 4]
    ],
    key=["H", "K", "V"],
    restore_value=["dl", "dq", "dk", "dv", "dg"],
)
@triton.jit(do_not_specialize=["T"])
def chunkwise_bwd_kernel_diag(
    do,
    q,
    k,
    v,
    g,
    l,
    llut,
    mask,
    dq,
    dk,
    dv,
    dg,
    dl,
    cu_seqlens,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    L: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    p_llut = tl.make_block_ptr(llut, (BT, BT), (BT, 1), (0, 0), (BT, BT), (1, 0))
    b_llut = tl.load(p_llut, boundary_check=(0, 1))
    i_t, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H

    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T

    o_i = tl.arange(0, BT)
    i_idx = o_i[:, None]
    j_idx = o_i[None, :]

    b_h_ptrs = l + ((bos + i_t * BT + i_idx) * H + i_h) * L + b_llut
    b_h = tl.load(b_h_ptrs, mask=i_idx >= j_idx)

    p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_q = tl.make_block_ptr(q + bos * K, (K, T), (1, K), (0, i_t * BT), (K, BT), (0, 1))
    p_k = tl.make_block_ptr(k + bos * K, (T, K), (K, 1), (i_t * BT, 0), (BT, K), (1, 0))
    p_v = tl.make_block_ptr(
        v + (bos * H + i_h) * V, (V, T), (1, H * V), (0, i_t * BT), (V, BT), (0, 1),
    )
    p_do = tl.make_block_ptr(
        do + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_t * BT, 0), (BT, V), (1, 0),
    )
    p_dg = tl.make_block_ptr(dg + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_dq = tl.make_block_ptr(
        dq + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT, 0), (BT, K), (1, 0),
    )
    p_dk = tl.make_block_ptr(
        dk + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT, 0), (BT, K), (1, 0),
    )
    p_dv = tl.make_block_ptr(
        dv + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_t * BT, 0), (BT, V), (1, 0),
    )

    b_g = tl.load(p_g, boundary_check=(0,))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_do = tl.load(p_do, boundary_check=(0, 1))
    b_dq = tl.load(p_dq, boundary_check=(0, 1))
    b_dk = tl.load(p_dk, boundary_check=(0, 1))
    b_dv = tl.load(p_dv, boundary_check=(0, 1))
    b_dg = tl.load(p_dg, boundary_check=(0,))

    b_s = (tl.dot(b_k, b_q)).to(b_q.dtype)
    m_t = i_t * BT + o_i < T
    b_a = tl.where((i_idx >= j_idx) & m_t[:, None] & m_t[None, :], tl.exp(b_g[:, None] - b_g[None, :]), 0)
    b_dv += tl.dot((b_s * tl.trans(b_a * b_h)).to(b_do.dtype), b_do)
    b_ds = tl.dot(b_do, b_v) * b_a
    b_dl = b_ds * tl.trans(b_s)
    b_dg += tl.sum(b_dl * b_h, axis=1)
    b_dg -= tl.sum(b_dl * b_h, axis=0)
    b_ds = (b_ds * b_h).to(b_k.dtype)
    b_dq += tl.dot(b_ds, b_k)
    b_dk += tl.trans(tl.dot(b_q, b_ds))

    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))

    num_intra_levels = (tl.log2(float(BT))).to(tl.int32) + 1

    for i in range(num_intra_levels):
        p_mask = tl.make_block_ptr(mask + i * (BT * BT), (BT, BT), (BT, 1), (0, 0), (BT, BT), (1, 0))
        b_mask = tl.load(p_mask, boundary_check=(0, 1))
        dl_i = tl.sum(tl.where(b_mask == 1, b_dl, 0), axis=1)
        p_dl_i = tl.make_block_ptr(dl + (bos * H + i_h) * L + i, (T,), (H * L,), (i_t * BT,), (BT,), (0,))
        tl.store(p_dl_i, dl_i, boundary_check=(0,))
