# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: tests/ops/ascend_ops/cv_operator/chunkwise_bwd_kernel_hdqgl/test_chunkwise_bwd_kernel_hdqgl_npu.py:30 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
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
    restore_value=["dq", "dg"],
)
@triton.jit(do_not_specialize=["T"])
def chunkwise_bwd_kernel_hdqgl(
    do,
    q,
    k,
    v,
    g,
    l,
    h_l,
    dq,
    dg,
    dl,
    ell,
    T,
    cu_seqlens,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    L: tl.constexpr,
    BT: tl.constexpr,
    NT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_nh = tl.program_id(0)
    i_n, i_h = i_nh // H, i_nh % H

    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T

    b_h = tl.zeros([V, K], dtype=tl.float32)

    num_intra_levels = (tl.log2(float(BT))).to(tl.int32) + 1

    for i_t in range(tl.cdiv(T, BT)):
        p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        if i_t & (1 << ell):
            p_do = tl.make_block_ptr(
                do + (bos * H + i_h) * V,
                (T, V),
                (H * V, 1),
                (i_t * BT, 0),
                (BT, V),
                (1, 0),
            )
            p_q = tl.make_block_ptr(
                q + bos * K, (T, K), (K, 1), (i_t * BT, 0), (BT, K), (1, 0),
            )
            p_l = tl.make_block_ptr(
                l + (bos * H + i_h) * L + num_intra_levels + ell,
                (T,),
                (H * L,),
                (i_t * BT,),
                (BT,),
                (0,),
            )
            p_dq = tl.make_block_ptr(
                dq + (bos * H + i_h) * K,
                (T, K),
                (H * K, 1),
                (i_t * BT, 0),
                (BT, K),
                (1, 0),
            )
            p_dg = tl.make_block_ptr(
                dg + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,),
            )
            p_dl = tl.make_block_ptr(
                dl + (bos * H + i_h) * L + num_intra_levels + ell,
                (T,),
                (H * L,),
                (i_t * BT,),
                (BT,),
                (0,),
            )
            p_h = tl.make_block_ptr(
                h_l + ((i_n * NT + i_t) * H + i_h) * K * V,
                (V, K),
                (1, V),
                (0, 0),
                (V, K),
                (0, 1),
            )

            b_do = tl.load(p_do, boundary_check=(0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_g = tl.load(p_g, boundary_check=(0,))
            b_l = tl.load(p_l, boundary_check=(0,))

            b_dlq = tl.exp(b_g)[:, None] * tl.dot(b_do, b_h.to(b_do.dtype))

            b_dl = tl.sum(b_dlq * b_q, axis=1)
            b_dg = b_l * b_dl

            tl.store(p_h, b_h, boundary_check=(0, 1))
            b_dq_old = tl.load(p_dq, boundary_check=(0, 1))
            tl.store(
                p_dq,
                (b_l[:, None] * b_dlq).to(p_dq.dtype.element_ty) + b_dq_old,
                boundary_check=(0, 1),
            )
            tl.store(p_dl, b_dl.to(p_dl.dtype.element_ty), boundary_check=(0,))
            b_dg_old = tl.load(p_dg, boundary_check=(0,))
            tl.store(
                p_dg, b_dg.to(p_dg.dtype.element_ty) + b_dg_old, boundary_check=(0,),
            )
            if ((i_t + 1) & (1 << ell)) == 0:
                b_h = tl.zeros([V, K], dtype=tl.float32)

        last_idx = min((i_t + 1) * BT, T) - 1
        b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
        b_h *= tl.exp(b_g_last)
        if (i_t & (1 << ell)) == 0:
            p_k = tl.make_block_ptr(
                k + bos * K, (T, K), (K, 1), (i_t * BT, 0), (BT, K), (1, 0),
            )
            p_v = tl.make_block_ptr(
                v + (bos * H + i_h) * V,
                (V, T),
                (1, H * V),
                (0, i_t * BT),
                (V, BT),
                (0, 1),
            )
            b_g = tl.load(p_g, boundary_check=(0,))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_v = tl.load(p_v, boundary_check=(0, 1))
            b_k = (b_k * tl.exp(b_g_last - b_g)[:, None]).to(b_k.dtype)
            b_h += tl.dot(b_v, b_k)
