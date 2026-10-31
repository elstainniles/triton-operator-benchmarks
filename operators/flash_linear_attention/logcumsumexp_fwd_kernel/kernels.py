# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: tests/ops/ascend_ops/cv_operator/chunk_abc/chunk_abc.py:108 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
# Source archive: dependencies and launch adaptation are not executable here.
# The complete original file and dependency snapshots are under upstream/.
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BT': BT}, num_warps=num_warps)
        for BT in [16, 32, 64]
        for num_warps in [2, 4, 8]
    ],
    key=['S'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def logcumsumexp_fwd_kernel(
    s,
    z,
    T,
    S: tl.constexpr,
    BT: tl.constexpr,
):
    i_bh = tl.program_id(0)
    o_i = tl.arange(0, BT)
    m_s = tl.where(o_i[:, None] >= o_i[None, :], 1., 0.)

    b_mp = tl.full([S], float('-inf'), dtype=tl.float32)
    b_zp = tl.zeros([S], dtype=tl.float32)
    for i_t in range(tl.cdiv(T, BT)):
        p_s = tl.make_block_ptr(s + i_bh * T*S, (T, S), (S, 1), (i_t * BT, 0), (BT, S), (1, 0))
        p_z = tl.make_block_ptr(z + i_bh * T*S, (T, S), (S, 1), (i_t * BT, 0), (BT, S), (1, 0))

        b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
        b_mc = tl.max(b_s, 0)
        b_mc = tl.maximum(b_mp, b_mc)
        b_zp = b_zp * exp(b_mp - b_mc)
        b_s = exp(b_s - b_mc)
        b_z = tl.dot(m_s, b_s, allow_tf32=False) + b_zp
        b_zc = tl.max(b_z, 0)
        b_mp = b_mc
        b_zp = b_zc
        b_z = log(tl.where(b_z != 0, b_z, 1e-20)) + b_mc
        tl.store(p_z, b_z.to(p_z.dtype.element_ty), boundary_check=(0, 1))
