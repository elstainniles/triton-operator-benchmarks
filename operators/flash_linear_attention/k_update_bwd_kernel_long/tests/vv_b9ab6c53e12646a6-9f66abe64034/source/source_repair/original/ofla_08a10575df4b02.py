# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import triton

import triton.language as tl

from ofla_74f55558e08fc1 import IS_AMD
from ofla_74f55558e08fc1 import autotune_cache_kwargs
from ofla_74f55558e08fc1 import get_multiprocessor_count
from ofla_74f55558e08fc1 import input_guard

NUM_WARPS_AUTOTUNE = [2, 4, 8, 16] if IS_AMD else [2, 4, 8, 16, 32]

@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(
    configs=[
        triton.Config({'BT': BT}, num_warps=w, num_stages=s)
        for w in NUM_WARPS_AUTOTUNE
        for s in [1, 2, 3]
        for BT in [2, 4, 8]
    ],
    key=['BD'],
    **autotune_cache_kwargs,
)
@triton.jit
def k_update_bwd_kernel_short(
    grad_out, k, a, ka,
    dk, da, dka,
    cu_seqlens,
    T, D,
    BT: tl.constexpr,
    BD: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_b, i_t_base = tl.program_id(0), tl.program_id(1) * BT

    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_b).to(tl.int32)
        eos = tl.load(cu_seqlens + i_b + 1).to(tl.int32)
        seq_len = eos - bos
    else:
        bos = i_b * T
        eos = (i_b + 1) * T
        seq_len = T

    t_vec = i_t_base + tl.arange(0, BT)
    mask_t = t_vec < seq_len
    global_t_vec = bos + t_vec

    o_d = tl.arange(0, BD)[None, :]
    m_d = o_d < D
    off = global_t_vec[:, None] * D + o_d

    b_go = tl.load(grad_out + off, mask=mask_t[:, None] & m_d, other=0.).to(tl.float32)
    b_k = tl.load(k + off, mask=mask_t[:, None] & m_d, other=0.).to(tl.float32)
    b_a = tl.load(a + off, mask=mask_t[:, None] & m_d, other=0.).to(tl.float32)
    b_ka = tl.load(ka + o_d, mask=m_d, eviction_policy='evict_last').to(tl.float32)  # [1, BD]

    dk_vec = b_go * (1 + (b_a - 1) * b_ka)
    da_vec = b_go * b_k * b_ka
    dka_vec = b_go * b_k * (b_a - 1)
    tl.store(dk + off, dk_vec.to(dk.dtype.element_ty), mask=mask_t[:, None] & m_d)
    tl.store(da + off, da_vec.to(da.dtype.element_ty), mask=mask_t[:, None] & m_d)
    tl.store(dka + off, dka_vec.to(dka.dtype.element_ty), mask=mask_t[:, None] & m_d)

@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=w, num_stages=s)
        for w in NUM_WARPS_AUTOTUNE
        for s in [1, 2, 3]
    ],
    key=['BD', 'BT'],
    **autotune_cache_kwargs,
)
@triton.jit
def k_update_bwd_kernel_long(
    grad_out, k, a, ka,
    dk, da, dka,
    cu_seqlens, chunk_indices,
    T, D,
    BD: tl.constexpr, BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_d, i_t_blk, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    if IS_VARLEN:
        i_n, i_t_blk = tl.load(chunk_indices + i_t_blk * 2).to(tl.int32), \
            tl.load(chunk_indices + i_t_blk * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        t_start = i_t_blk * BT
        t_end = tl.minimum(t_start + BT, eos - bos)
    else:
        bos = i_b * T
        eos = (i_b + 1) * T
        t_start = i_t_blk * BT
        t_end = tl.minimum(t_start + BT, T)

    o_d = i_d * BD + tl.arange(0, BD)
    m_d = o_d < D

    for t in range(t_start, t_end):
        global_t = bos + t
        off = global_t * D + o_d

        b_go = tl.load(grad_out + off, mask=m_d, other=0.).to(tl.float32)
        b_k = tl.load(k + off, mask=m_d, other=0.).to(tl.float32)
        b_a = tl.load(a + off, mask=m_d, other=0.).to(tl.float32)
        b_ka = tl.load(ka + o_d, mask=m_d, eviction_policy='evict_last').to(tl.float32)

        tl.store(dk + off, (b_go * (1 + (b_a - 1) * b_ka)).to(dk.dtype.element_ty), mask=m_d)
        tl.store(da + off, (b_go * b_k * b_ka).to(da.dtype.element_ty), mask=m_d)
        tl.store(dka + off, (b_go * b_k * (b_a - 1)).to(dka.dtype.element_ty), mask=m_d)
