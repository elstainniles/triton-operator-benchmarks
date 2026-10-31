# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: tests/ops/ascend_ops/cv_operator/chunkwise_bwd_kernel_dhg/test_chunkwise_bwd_kernel_dhg_npu.py:33 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
# Source archive: dependencies and launch adaptation are not executable here.
# The complete original file and dependency snapshots are under upstream/.
import triton
import triton.language as tl


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[
        triton.Config({"BK": BK}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64, 128]
        for num_warps in [4]
        for num_stages in [2, 3, 4]
    ],
    key=["H", "K", "V"],
    restore_value=["dh", "dg_last"],
)
@triton.jit(do_not_specialize=["T"])
def chunkwise_bwd_kernel_dhg(
    do,
    q,
    g,
    l,
    h_l,
    dh,
    dg_last,
    ell,
    T,
    cu_seqlens,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    L: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    NT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_k = tl.program_id(0)
    i_nh = tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H

    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T

    b_dh = tl.zeros([BK, V], dtype=tl.float32)

    num_intra_levels = (tl.log2(float(BT))).to(tl.int32) + 1

    for i_t in range(tl.cdiv(T, BT) - 1, -1, -1):
        p_dh = tl.make_block_ptr(
            dh + ((i_n * NT + i_t) * H + i_h) * K * V,
            (K, V),
            (V, 1),
            (i_k * BK, 0),
            (BK, V),
            (1, 0),
        )
        b_dh_old = tl.load(p_dh, boundary_check=(0, 1))

        if (i_t & (1 << ell)) == 0:
            tl.store(
                p_dh, b_dh.to(p_dh.dtype.element_ty) + b_dh_old, boundary_check=(0, 1),
            )
            if i_t > 0 and ((i_t - 1) & (1 << ell)) > 0:
                b_dh = tl.zeros([BK, V], dtype=tl.float32)
        if i_t & (1 << ell):
            p_h = tl.make_block_ptr(
                h_l + ((i_n * NT + i_t) * H + i_h) * K * V,
                (K, V),
                (V, 1),
                (i_k * BK, 0),
                (BK, V),
                (1, 0),
            )

            b_h = tl.load(p_h, boundary_check=(0, 1))
            p_dg_last = dg_last + i_n * NT * H + i_t * H + i_h
            tl.atomic_add(p_dg_last, tl.sum(b_h * (b_dh + b_dh_old)))
        last_idx = min((i_t + 1) * BT, T) - 1
        b_g_last = tl.exp(tl.load(g + bos * H + last_idx * H + i_h))
        b_dh *= b_g_last
        if i_t & (1 << ell):
            p_g = tl.make_block_ptr(
                g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,),
            )
            p_q = tl.make_block_ptr(
                q + bos * K, (K, T), (1, K), (i_k * BK, i_t * BT), (BK, BT), (0, 1),
            )
            p_do = tl.make_block_ptr(
                do + (bos * H + i_h) * V,
                (T, V),
                (H * V, 1),
                (i_t * BT, 0),
                (BT, V),
                (1, 0),
            )
            p_l = tl.make_block_ptr(
                l + (bos * H + i_h) * L + num_intra_levels + ell,
                (T,),
                (H * L,),
                (i_t * BT,),
                (BT,),
                (0,),
            )
            b_l = tl.load(p_l, boundary_check=(0,))
            b_g = tl.load(p_g, boundary_check=(0,))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_q = (b_q * (tl.exp(b_g) * b_l)[None, :]).to(b_q.dtype)
            b_do = tl.load(p_do, boundary_check=(0, 1))

            b_s = tl.dot(b_q, b_do).to(b_q.dtype)
            b_dh += b_s
