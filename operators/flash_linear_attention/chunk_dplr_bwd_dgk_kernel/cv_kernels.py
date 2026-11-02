# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Fixed CV launch adapter; JIT signatures and bodies match upstream/.
# Autotune/heuristics are bound explicitly by cases-cv.json.
# Generated from exact archived source; function bodies unchanged.
import triton
import triton.language as tl
import math

@triton.jit(do_not_specialize=['T'])
def chunk_dplr_bwd_dgk_kernel(dgk, dgk_offset, dgk_last, dgk_output, cu_seqlens, chunk_indices, T, H: tl.constexpr, K: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, IS_VARLEN: tl.constexpr):
    i_t, i_k, i_bh = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    i_b, i_h = (i_bh // H, i_bh % H)
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = (tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32))
        bos, eos = (tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32))
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = (i_b * NT + i_t).to(tl.int32)
        bos, eos = ((i_b * T).to(tl.int32), (i_b * T + T).to(tl.int32))
    stride_qk = H * K
    dgk += (bos * H + i_h) * K
    dgk_offset += (bos * H + i_h) * K
    dgk_last += (i_tg * H + i_h) * K
    dgk_output += (bos * H + i_h) * K
    p_dgk_last = dgk_last + tl.arange(0, BK) + i_k * BK
    m_k = tl.arange(0, BK) + i_k * BK < K
    b_dgk_last = tl.load(p_dgk_last, mask=m_k, other=0)
    p_dgk_offset = tl.make_block_ptr(dgk_offset, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dgk = tl.make_block_ptr(dgk, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    b_dgk = tl.load(p_dgk, boundary_check=(0, 1))
    b_dgk_offset = tl.load(p_dgk_offset, boundary_check=(0, 1))
    b_dgk_cumsum = tl.cumsum(b_dgk, 0, reverse=True)
    b_dgk_cumsum += b_dgk_last[None, :]
    b_dgk_cumsum -= b_dgk_offset
    p_dgk_output = tl.make_block_ptr(dgk_output, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    tl.store(p_dgk_output, b_dgk_cumsum.to(p_dgk_output.dtype.element_ty), boundary_check=(0, 1))
