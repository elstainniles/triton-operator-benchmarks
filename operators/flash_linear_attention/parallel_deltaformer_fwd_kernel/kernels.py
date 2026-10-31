# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: fla/ops/deltaformer/parallel.py:137 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
# Source archive: dependencies and launch adaptation are not executable here.
# The complete original file and dependency snapshots are under upstream/.
import triton
import triton.language as tl


@triton.autotune(configs=_config_deltaformer(), key=['C', 'D'])
@triton.jit
def parallel_deltaformer_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    u_ptr,
    w_ptr,
    lse_ptr,
    beta_ptr,
    H,
    T,
    C,
    D: tl.constexpr,
    qk_scale: float,
    BLOCK_C: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid_c = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)

    rowid_block = tl.arange(0, BLOCK_C) + pid_c * BLOCK_C
    colid_block = tl.arange(0, BLOCK_T)

    rowmax = tl.zeros([BLOCK_C], dtype=tl.float32) - float('inf')
    rowsum = tl.zeros([BLOCK_C], dtype=tl.float32) + 1
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
        qk = tl.dot(q, k) * qk_scale

        if kv_i >= T - C:
            mask = (T - C - kv_i + rowid_block[:, None] - colid_block[None, :] < 1)
            qk = tl.where(mask, -1e6, qk)

        rowmax_i = tl.maximum(rowmax, tl.max(qk, axis=1))
        qk -= rowmax_i[:, None]
        p = tl.math.exp2(qk)

        rowsum_i = tl.sum(p, axis=1)
        alpha = tl.math.exp2(rowmax - rowmax_i)
        rowsum = rowsum * alpha + rowsum_i
        acc = acc * alpha[:, None]
        rowmax = rowmax_i

        if kv_i < T - C:
            u_blk_ptr = tl.make_block_ptr(
                base=u_ptr + pid_h * D,
                shape=(T, D),
                strides=(H * D, 1),
                offsets=(kv_i, 0),
                block_shape=(BLOCK_T, D),
                order=(1, 0),
            )
            u = tl.load(u_blk_ptr, boundary_check=(0,))
            acc = tl.dot(p.to(u_ptr.dtype.element_ty), u, acc)

    lse = rowmax + tl.math.log2(rowsum)
    lse_block_ptr = lse_ptr + pid_h + rowid_block * H
    lse_mask = rowid_block < C
    tl.store(lse_block_ptr, lse, mask=lse_mask)

    v_ptr = tl.make_block_ptr(
        base=v_ptr + pid_h * D,
        shape=(C, D),
        strides=(H * D, 1),
        offsets=(pid_c * BLOCK_C, 0),
        block_shape=(BLOCK_C, D),
        order=(1, 0),
    )
    acc = acc / rowsum[:, None]

    beta_ptr = tl.make_block_ptr(
        base=beta_ptr + pid_h,
        shape=(C,),
        strides=(H,),
        offsets=(pid_c * BLOCK_C,),
        block_shape=(BLOCK_C,),
        order=(0,),
    )
    beta = tl.load(beta_ptr, boundary_check=(0,))
    acc = acc * beta[:, None]

    v = tl.load(v_ptr, boundary_check=(0,))
    u = v - acc.to(v_ptr.dtype.element_ty)
    u_block_ptr = tl.make_block_ptr(
        base=u_ptr + pid_h * D,
        shape=(T, D),
        strides=(H * D, 1),
        offsets=(T - C + pid_c * BLOCK_C, 0),
        block_shape=(BLOCK_C, D),
        order=(1, 0),
    )
    tl.store(u_block_ptr, u, boundary_check=(0, 1))

    for kv_i in range(T - C, T, BLOCK_T):
        k_blk_ptr = tl.make_block_ptr(
            base=k_ptr + pid_h * D,
            shape=(D, T),
            strides=(1, H * D),
            offsets=(0, kv_i),
            block_shape=(D, BLOCK_T),
            order=(0, 1),
        )
        k = tl.load(k_blk_ptr, boundary_check=(1,))
        qk = tl.dot(q, k) * qk_scale

        mask = (T - C - kv_i + rowid_block[:, None] - colid_block[None, :] < 1)
        qk -= rowmax[:, None]
        p = tl.math.exp2(qk) / rowsum[:, None]
        p = tl.where(mask, 0, p)
        w_blk_ptr = tl.make_block_ptr(
            base=w_ptr + pid_h * C,
            shape=(C, C),
            strides=(H * C, 1),
            offsets=(pid_c * BLOCK_C, kv_i - (T - C)),
            block_shape=(BLOCK_C, BLOCK_T),
            order=(1, 0),
        )
        tl.store(w_blk_ptr, p.to(w_ptr.dtype.element_ty), boundary_check=(0, 1))
