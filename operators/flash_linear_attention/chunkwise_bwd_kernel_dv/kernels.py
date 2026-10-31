# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: tests/ops/ascend_ops/cv_operator/chunkwise_bwd_kernel_dv/test_chunkwise_bwd_kernel_dv_npu.py:25 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
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
    restore_value=["dv"],
)
@triton.jit(do_not_specialize=["T"])
def chunkwise_bwd_kernel_dv(
    dh,
    k,
    g,
    dv,
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

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T

    p_dh = tl.make_block_ptr(
        dh + ((i_n * NT + i_t) * H + i_h) * K * V,
        (K, V),
        (V, 1),
        (0, 0),
        (K, V),
        (1, 0),
    )
    p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_k = tl.make_block_ptr(k + bos * K, (T, K), (K, 1), (i_t * BT, 0), (BT, K), (1, 0))
    p_dv = tl.make_block_ptr(
        dv + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_t * BT, 0), (BT, V), (1, 0),
    )

    last_idx = min((i_t + 1) * BT, T) - 1
    b_g_last = tl.load(g + bos * H + last_idx * H + i_h)

    b_dh = tl.load(p_dh, boundary_check=(0, 1))
    b_g = tl.load(p_g, boundary_check=(0,))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_dv = tl.where(m_t, tl.exp(-b_g + b_g_last), 0)[:, None] * tl.dot(b_k, b_dh).to(b_k.dtype)
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
