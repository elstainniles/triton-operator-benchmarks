# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Fixed CV launch adapter; JIT signatures and bodies match upstream/.
# Autotune/heuristics are bound explicitly by cases-cv.json.
# Generated from exact archived source; function bodies unchanged.
import triton
import triton.language as tl
import math

@triton.jit(do_not_specialize=['T'])
def chunk_rwkv6_fwd_cumsum_kernel(s, oi, oe, scale, cu_seqlens, chunk_indices, T, H: tl.constexpr, S: tl.constexpr, BT: tl.constexpr, BS: tl.constexpr, HAS_SCALE: tl.constexpr, IS_VARLEN: tl.constexpr):
    i_s, i_t, i_bh = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    i_b, i_h = (i_bh // H, i_bh % H)
    if IS_VARLEN:
        i_n, i_t = (tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32))
        bos, eos = (tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32))
        T = eos - bos
    else:
        bos, eos = (i_b * T, i_b * T + T)
    o_i = tl.arange(0, BT)
    m_i = tl.where(o_i[:, None] >= o_i[None, :], 1.0, 0.0).to(tl.float32)
    m_e = tl.where(o_i[:, None] > o_i[None, :], 1.0, 0.0).to(tl.float32)
    p_s = tl.make_block_ptr(s + (bos * H + i_h) * S, (T, S), (H * S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    p_oi = tl.make_block_ptr(oi + (bos * H + i_h) * S, (T, S), (H * S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    p_oe = tl.make_block_ptr(oe + (bos * H + i_h) * S, (T, S), (H * S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
    b_oi = tl.dot(m_i, b_s)
    b_oe = tl.dot(m_e, b_s)
    if HAS_SCALE:
        b_oi = b_oi * scale
        b_oe = b_oe * scale
    tl.store(p_oi, b_oi.to(p_oi.dtype.element_ty, fp_downcast_rounding='rtne'), boundary_check=(0, 1))
    tl.store(p_oe, b_oe.to(p_oe.dtype.element_ty, fp_downcast_rounding='rtne'), boundary_check=(0, 1))
