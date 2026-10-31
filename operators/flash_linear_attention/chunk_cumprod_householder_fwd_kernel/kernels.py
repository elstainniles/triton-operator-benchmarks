# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: tests/ops/ascend_ops/cv_operator/chunk_cumprod_householder_fwd_kernel/test_chunk_cumprod_householder_fwd_kernel_npu.py:116 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
# Source archive: dependencies and launch adaptation are not executable here.
# The complete original file and dependency snapshots are under upstream/.
import triton
import triton.language as tl


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit
def chunk_cumprod_householder_fwd_kernel(
    k,
    k_new,
    w1,
    w2,
    hc_suffix,
    hc_whole,
    cu_seqlens,
    split_indices,
    chunk_offsets,
    split_offsets,
    BT: tl.constexpr,  # small chunk size
    K: tl.constexpr,
    H: tl.constexpr,
    BK: tl.constexpr,
    T: tl.constexpr,
    S: tl.constexpr,  # split size, aka large chunk size
    IS_VARLEN: tl.constexpr,
):
    i_ss, i_h = tl.program_id(0), tl.program_id(1)

    if IS_VARLEN:
        i_n, i_s = tl.load(split_indices + i_ss * 2).to(tl.int32), tl.load(split_indices + i_ss * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NS = tl.cdiv(T, S)

        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
        boh_large = tl.load(split_offsets + i_n).to(tl.int32)
    else:
        NS = tl.cdiv(T, S)
        i_n, i_s = i_ss // NS, i_ss % NS
        bos, eos = i_n * T, i_n * T + T

        boh = i_n * tl.cdiv(T, BT)
        boh_large = i_n * tl.cdiv(T, S)

    NT_small = tl.cdiv(min(S, T-i_s*S), BT)
    stride_h = H*K*K

    # offset calculations
    hc_whole += ((boh_large + i_s) * H + i_h) * K * K
    hc_suffix += ((boh + tl.cdiv(i_s * S, BT)) * H + i_h) * K * K

    k += (bos * H + i_h) * K
    k_new += (bos * H + i_h) * K
    w1 += (bos * H + i_h) * K
    w2 += (bos * H + i_h) * K

    b_h = tl.zeros([BK, BK], dtype=tl.float32)
    for i_t_small in range(NT_small-1, -1, -1):
        p_hc_suffix = tl.make_block_ptr(hc_suffix + i_t_small * stride_h, (K, K), (K, 1), (0, 0), (BK, BK), (1, 0))
        tl.store(p_hc_suffix, b_h.to(hc_suffix.dtype.element_ty), boundary_check=(0, 1))
        p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_s * S + i_t_small * BT, 0), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_k = (b_k - tl.dot(b_k, tl.trans(b_h.to(b_k.dtype))))
        p_w1 = tl.make_block_ptr(w1, (K, T), (1, H*K), (0, i_s * S + i_t_small * BT), (BK, BT), (0, 1))
        p_w2 = tl.make_block_ptr(w2, (T, K), (H*K, 1), (i_s * S + i_t_small * BT, 0), (BT, BK), (1, 0))
        b_w1 = tl.load(p_w1, boundary_check=(0, 1))
        b_w2 = tl.load(p_w2, boundary_check=(0, 1))
        b_v_new = (b_w1 - tl.dot(b_h.to(b_w1.dtype), b_w1)).to(b_w2.dtype)
        b_h += tl.dot(b_v_new, b_w2)
        p_k_new = tl.make_block_ptr(k_new, (T, K), (H*K, 1), (i_s * S + i_t_small * BT, 0), (BT, BK), (1, 0))
        tl.store(p_k_new, b_k.to(k_new.dtype.element_ty), boundary_check=(0, 1))

    p_hc_whole = tl.make_block_ptr(hc_whole, (K, K), (K, 1), (0, 0), (BK, BK), (1, 0))
    tl.store(p_hc_whole, b_h.to(hc_whole.dtype.element_ty), boundary_check=(0, 1))
