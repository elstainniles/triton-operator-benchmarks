# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: tests/ops/ascend_ops/cv_operator/test_fused_recurrent_rwkv6_bwd_kernel_dw/test_fused_recurrent_rwkv6_bwd_kernel_dw_cuda.py:22 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
# Source archive: dependencies and launch adaptation are not executable here.
# The complete original file and dependency snapshots are under upstream/.
import triton
import triton.language as tl


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BT': BT, 'BK': BK}, num_warps=num_warps)
        for BT in [16, 32, 64]
        for BK in [32, 64]
        for num_warps in [1, 2, 4, 8]
    ],
    key=['K'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_rwkv6_bwd_kernel_dw(
    q,
    k,
    dq,
    dk,
    dw,
    cu_seqlens,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    REVERSE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_k, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
    T = eos - bos
    NT = tl.cdiv(T, BT)

    o_i = tl.arange(0, BT)
    m_i = tl.where(o_i[:, None] >= o_i[None, :], 1., 0.) if not REVERSE else tl.where(o_i[:, None] <= o_i[None, :], 1., 0.)

    b_z = tl.zeros([BK], dtype=tl.float32)

    i_t = 0 if not REVERSE else NT - 1
    for _ in range(NT):
        p_q = tl.make_block_ptr(q + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT + 1, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k + (bos*H + i_h) * K, (T-1, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dq = tl.make_block_ptr(dq + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT + 1, i_k * BK), (BT, BK), (1, 0))
        p_dk = tl.make_block_ptr(dk + (bos*H + i_h) * K, (T-1, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dw = tl.make_block_ptr(dw + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32)
        b_dq = tl.load(p_dq, boundary_check=(0, 1)).to(tl.float32)
        b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
        b_dk = tl.load(p_dk, boundary_check=(0, 1)).to(tl.float32)
        b_dw = (b_q * b_dq * scale) - b_k * b_dk
        b_c = b_z[None, :] + tl.dot(m_i, b_dw, allow_tf32=False)
        tl.store(p_dw, b_c.to(p_dw.dtype.element_ty), boundary_check=(0, 1))
        if i_t >= 0:
            b_z += tl.sum(b_dw, 0)

        i_t += (1 if not REVERSE else -1)
