# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: tests/ops/ascend_ops/vv_operator/k_update_bwd_kernel_short/test_k_update_bwd_kernel_short_npu.py:23 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
# Source archive: dependencies and launch adaptation are not executable here.
# The complete original file and dependency snapshots are under upstream/.
import triton
import triton.language as tl


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(
    configs=[
        triton.Config({'BT': BT}, num_warps=w, num_stages=s)
        for w in NUM_WARPS_AUTOTUNE
        for s in [1, 2, 3]
        for BT in [2, 4, 8]
    ],
    key=['BD'],
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
    b_ka = tl.load(ka + o_d, mask=m_d, eviction_policy='evict_last').to(tl.float32)

    dk_vec = b_go * (1 + (b_a - 1) * b_ka)
    da_vec = b_go * b_k * b_ka
    dka_vec = b_go * b_k * (b_a - 1)
    tl.store(dk + off, dk_vec.to(dk.dtype.element_ty), mask=mask_t[:, None] & m_d)
    tl.store(da + off, da_vec.to(da.dtype.element_ty), mask=mask_t[:, None] & m_d)
    tl.store(dka + off, dka_vec.to(dka.dtype.element_ty), mask=mask_t[:, None] & m_d)
