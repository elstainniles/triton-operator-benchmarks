# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: fla/ops/deltaformer/parallel.py:278 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
# Source archive: dependencies and launch adaptation are not executable here.
# The complete original file and dependency snapshots are under upstream/.
import triton
import triton.language as tl


@triton.autotune(configs=_config_deltaformer(), key=['C', 'D'])
@triton.jit
def parallel_deltaformer_bwd_kernel_u(
    o_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    lse_ptr,
    beta_ptr,
    H,
    T,
    C,
    D: tl.constexpr,
    fa_scale,
    BLOCK_C: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid_c = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)

    acc = tl.zeros([BLOCK_C, D], dtype=tl.float32)

    q_blk_ptr = tl.make_block_ptr(
        base=q_ptr + pid_h * D,
        shape=(C, D),
        strides=(H * D, 1),
        offsets=(pid_c * BLOCK_C, 0),
        block_shape=(BLOCK_C, D),
        order=(1, 0),
    )
    q = tl.load(q_blk_ptr, boundary_check=(0,))

    for kv_i in range(0, T, BLOCK_T):
        k_blk_ptr = tl.make_block_ptr(
            base=k_ptr + pid_h * D,
            shape=(D, T),
            strides=(1, H * D),
            offsets=(0, kv_i),
            block_shape=(D, BLOCK_T),
            order=(0, 1),
        )
        k = tl.load(k_blk_ptr, boundary_check=(1,))
        qk = tl.dot(q, k) * fa_scale

        lse_blk_ptr = tl.make_block_ptr(
            base=lse_ptr + pid_h,
            shape=(T,),
            strides=(H,),
            offsets=(kv_i,),
            block_shape=(BLOCK_T,),
            order=(0,),
        )
        lse = tl.load(lse_blk_ptr, boundary_check=(0,))
        beta_blk_ptr = tl.make_block_ptr(
            base=beta_ptr + pid_h,
            shape=(T,),
            strides=(H,),
            offsets=(kv_i,),
            block_shape=(BLOCK_T,),
            order=(0,),
        )
        beta = tl.load(beta_blk_ptr, boundary_check=(0,))

        p = tl.math.exp2(qk - lse[None, :]) * beta[None, :]

        v_blk_ptr = tl.make_block_ptr(
            base=v_ptr + pid_h * D,
            shape=(T, D),
            strides=(H * D, 1),
            offsets=(kv_i, 0),
            block_shape=(BLOCK_T, D),
            order=(1, 0),
        )
        v = tl.load(v_blk_ptr, boundary_check=(0,))
        acc = tl.dot(p.to(v_ptr.dtype.element_ty), v, acc)

    o_blk_ptr = tl.make_block_ptr(
        base=o_ptr + pid_h * D,
        shape=(C, D),
        strides=(H * D, 1),
        offsets=(pid_c * BLOCK_C, 0),
        block_shape=(BLOCK_C, D),
        order=(1, 0),
    )
    tl.store(o_blk_ptr, acc.to(o_ptr.dtype.element_ty), boundary_check=(0,))
