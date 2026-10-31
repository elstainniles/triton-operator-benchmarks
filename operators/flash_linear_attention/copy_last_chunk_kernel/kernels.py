# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: fla/ops/log_linear_attn/chunk.py:829 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
# Source archive: dependencies and launch adaptation are not executable here.
# The complete original file and dependency snapshots are under upstream/.
import triton
import triton.language as tl


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    },
)
@triton.jit(do_not_specialize=["T"])
def copy_last_chunk_kernel(
    q,
    k,
    v,
    g,
    level_scales,
    cu_seqlens,
    q_prev,
    k_prev,
    v_prev,
    g_prev,
    level_scales_prev,
    offsets,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    L: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    # parallel over sequences and heads
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

    seq_offset = (T // BT) * BT

    p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (seq_offset,), (BT,), (0,))
    p_q = tl.make_block_ptr(
        q + bos * K, (T, K), (K, 1), (seq_offset, 0), (BT, K), (1, 0),
    )
    p_k = tl.make_block_ptr(
        k + bos * K, (T, K), (K, 1), (seq_offset, 0), (BT, K), (1, 0),
    )
    p_v = tl.make_block_ptr(
        v + (bos * H + i_h) * V,
        (T, V),
        (H * V, 1),
        (seq_offset, 0),
        (BT, V),
        (1, 0),
    )
    p_g_prev = tl.make_block_ptr(
        g_prev + i_n * BT * H + i_h, (BT,), (H,), (0,), (BT,), (0,),
    )
    p_q_prev = tl.make_block_ptr(
        q_prev + i_n * BT * K, (BT, K), (K, 1), (0, 0), (BT, K), (1, 0),
    )
    p_k_prev = tl.make_block_ptr(
        k_prev + i_n * BT * K, (BT, K), (K, 1), (0, 0), (BT, K), (1, 0),
    )
    p_v_prev = tl.make_block_ptr(
        v_prev + (i_n * BT * H + i_h) * V, (BT, V), (H * V, 1), (0, 0), (BT, V), (1, 0),
    )

    tl.store(p_g_prev, tl.load(p_g, boundary_check=(0,)), boundary_check=(0,))
    tl.store(p_q_prev, tl.load(p_q, boundary_check=(0, 1)), boundary_check=(0, 1))
    tl.store(p_k_prev, tl.load(p_k, boundary_check=(0, 1)), boundary_check=(0, 1))
    tl.store(p_v_prev, tl.load(p_v, boundary_check=(0, 1)), boundary_check=(0, 1))

    for i in range(L):
        p_l = tl.make_block_ptr(
            level_scales + (bos * H + i_h) * L,
            (T, L),
            (H * L, 1),
            (seq_offset, i),
            (BT, 1),
            (1, 0),
        )
        p_l_prev = tl.make_block_ptr(
            level_scales_prev + (i_n * BT * H + i_h) * L,
            (BT, L),
            (H * L, 1),
            (0, i),
            (BT, 1),
            (1, 0),
        )
        tl.store(p_l_prev, tl.load(p_l, boundary_check=(0,)), boundary_check=(0,))
