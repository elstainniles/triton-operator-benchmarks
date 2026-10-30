# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: tests/ops/ascend_ops/cv_operator/chunkwise_bwd_kernel_dkg/test_chunkwise_bwd_kernel_dkg_npu.py:35 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
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
    restore_value=["dk", "dg", "dg_last"],
)
@triton.jit(do_not_specialize=["T"])
def chunkwise_bwd_kernel_dkg(
    dh,
    k,
    v,
    g,
    dg_last,
    dk,
    dg,
    cu_seqlens,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    L: tl.constexpr,
    BT: tl.constexpr,
    NT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
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
    o_t = i_t * BT + o_i
    m_t = o_t < T

    p_dh = tl.make_block_ptr(
        dh + ((i_n * NT + i_t) * H + i_h) * K * V,
        (V, K),
        (1, V),
        (0, 0),
        (V, K),
        (0, 1),
    )
    p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_k = tl.make_block_ptr(k + bos * K, (T, K), (K, 1), (i_t * BT, 0), (BT, K), (1, 0))
    p_v = tl.make_block_ptr(
        v + (bos * H + i_h) * V,
        (T, V),
        (H * V, 1),
        (i_t * BT, 0),
        (BT, V),
        (1, 0),
    )
    p_dk = tl.make_block_ptr(
        dk + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT, 0), (BT, K), (1, 0),
    )
    p_dg = tl.make_block_ptr(dg + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))

    b_dh = tl.load(p_dh, boundary_check=(0, 1))
    b_g = tl.load(p_g, boundary_check=(0,))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    last_idx = min((i_t + 1) * BT, T) - 1
    b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
    p_dg_last = dg_last + i_n * NT * H + i_t * H + i_h
    b_dg_last = tl.load(p_dg_last)

    b_dg_last *= tl.exp(b_g_last)
    b_dk = tl.where(m_t, exp(b_g_last - b_g), 0)[:, None] * tl.dot(b_v, b_dh).to(b_v.dtype)
    b_dg = tl.load(p_dg, boundary_check=(0,))
    b_dg -= tl.sum(b_k * b_dk, axis=1)
    b_dg_last += tl.sum(b_dk * b_k)

    b_dg = tl.where(o_i < BT - 1, b_dg, b_dg + b_dg_last)

    tl.store(p_dg, b_dg, boundary_check=(0,))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
