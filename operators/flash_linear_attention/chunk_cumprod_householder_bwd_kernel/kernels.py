# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: tests/ops/ascend_ops/cv_operator/chunk_cumprod_householder_bwd_kernel/test_chunk_cumprod_householder_bwd_kernel_npu.py:37 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
# Source archive: dependencies and launch adaptation are not executable here.
# The complete original file and dependency snapshots are under upstream/.
import triton
import triton.language as tl


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit()
def chunk_cumprod_householder_bwd_kernel(
    hc_suffix, dhc_whole,
    k, dk, w1, w2, dw1, dw2, dk_new,
    cu_seqlens, split_indices, chunk_offsets, split_offsets,
    BT: tl.constexpr,  # previous small chunk size
    K: tl.constexpr,
    BK: tl.constexpr,
    T: tl.constexpr,
    S: tl.constexpr,
    G: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_ss, i_hq = tl.program_id(0), tl.program_id(1)
    i_h = i_hq // G

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

    # offset calculations
    dhc_whole += ((boh_large + i_s) * HQ + i_hq) * K * K
    hc_suffix += ((boh + tl.cdiv(i_s * S, BT)) * H + i_h) * K * K
    k += (bos * H + i_h) * K
    w1 += (bos * H + i_h) * K
    w2 += (bos * H + i_h) * K
    dw1 += (bos * HQ + i_hq) * K
    dw2 += (bos * HQ + i_hq) * K

    # dh += ((boh + tl.cdiv(i_s * S, BT)) * HQ + i_hq) * K * K
    dk += (bos * HQ + i_hq) * K
    dk_new += (bos * HQ + i_hq) * K

    stride_h = H * K * K
    NT_small = tl.cdiv(min(S, T-i_s*S), BT)
    p_dhc_whole = tl.make_block_ptr(dhc_whole, (K, K), (K, 1), (0, 0), (BK, BK), (1, 0))
    b_dhc = tl.zeros([BK, BK], dtype=tl.float32)
    b_dhc += tl.load(p_dhc_whole, boundary_check=(0, 1))

    # calculate dh
    for i_t_small in range(0, NT_small):
        p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_s*S + i_t_small*BT, 0), (BT, BK), (1, 0))
        p_dk = tl.make_block_ptr(dk, (T, K), (HQ*K, 1), (i_s*S + i_t_small*BT, 0), (BT, BK), (1, 0))
        p_dk_new = tl.make_block_ptr(dk_new, (T, K), (HQ*K, 1), (i_s*S + i_t_small*BT, 0), (BT, BK), (1, 0))
        p_hc = tl.make_block_ptr(hc_suffix + i_t_small * stride_h, (K, K), (K, 1), (0, 0), (BK, BK), (1, 0))

        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_dk = tl.load(p_dk, boundary_check=(0, 1))

        p_w1 = tl.make_block_ptr(w1, (T, K), (H*K, 1), (i_s*S + i_t_small*BT, 0), (BT, BK), (1, 0))
        p_w2 = tl.make_block_ptr(w2, (T, K), (H*K, 1), (i_s*S + i_t_small*BT, 0), (BT, BK), (1, 0))

        b_w1 = tl.load(p_w1, boundary_check=(0, 1))
        b_w2 = tl.load(p_w2, boundary_check=(0, 1))
        b_hc = tl.load(p_hc, boundary_check=(0, 1))

        b_dk_new = b_dk - tl.dot(b_dk.to(b_hc.dtype), b_hc)
        tl.store(p_dk_new, b_dk_new.to(dk_new.dtype.element_ty), boundary_check=(0, 1))

        b_dh = b_dhc - tl.dot(tl.trans(b_hc), b_dhc.to(b_hc.dtype))
        b_dw2 = tl.dot(b_w1, b_dh.to(b_w1.dtype))
        b_dw1 = tl.dot(b_w2, tl.trans(b_dh.to(b_w2.dtype)))

        p_dw1 = tl.make_block_ptr(dw1, (T, K), (HQ*K, 1), (i_s*S + i_t_small*BT, 0), (BT, BK), (1, 0))
        p_dw2 = tl.make_block_ptr(dw2, (T, K), (HQ*K, 1), (i_s*S + i_t_small*BT, 0), (BT, BK), (1, 0))

        tl.store(p_dw1, b_dw1.to(dw1.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dw2, b_dw2.to(dw2.dtype.element_ty), boundary_check=(0, 1))

        b_dhc = b_dhc - tl.dot(tl.dot(b_dhc.to(b_w2.dtype), tl.trans(b_w2)).to(b_w1.dtype), b_w1)
        b_dhc -= tl.dot(tl.trans(b_dk).to(b_k.dtype), b_k)
