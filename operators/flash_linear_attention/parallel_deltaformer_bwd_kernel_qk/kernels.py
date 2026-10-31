# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: fla/ops/deltaformer/parallel.py:459 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
# Source archive: dependencies and launch adaptation are not executable here.
# The complete original file and dependency snapshots are under upstream/.
import triton
import triton.language as tl


@triton.autotune(configs=[triton.Config({'BLOCK_C': BC}, num_stages=ns, num_warps=nw)
                          for BC in [64, 32]
                          for ns in [4, 3]
                          for nw in [4]], key=['T', 'D'])
@triton.jit
def parallel_deltaformer_bwd_kernel_qk(
    grad_q_ptr,
    grad_k_ptr,
    q_ptr,
    k_ptr,
    grad_v_ptr,
    u_ptr,
    lse_ptr,
    beta_ptr,
    row_dot_ptr,
    H,
    T,
    D: tl.constexpr,
    fa_scale: tl.constexpr,
    qk_scale: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_c = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    block_i = tl.arange(0, BLOCK_C)

    acc = tl.zeros([BLOCK_C, D], dtype=tl.float32)

    k_row_blk_ptr = tl.make_block_ptr(
        base=q_ptr + pid_h * D,
        shape=(T, D),
        strides=(H * D, 1),
        offsets=(pid_c * BLOCK_C, 0),
        block_shape=(BLOCK_C, D),
        order=(1, 0),
    )
    k_row = tl.load(k_row_blk_ptr, boundary_check=(0,))
    lse_blk_ptr = tl.make_block_ptr(
        base=lse_ptr + pid_h,
        shape=(T,),
        strides=(H,),
        offsets=(pid_c * BLOCK_C,),
        block_shape=(BLOCK_C,),
        order=(0,),
    )
    lse = tl.load(lse_blk_ptr, boundary_check=(0,))
    beta_blk_ptr = tl.make_block_ptr(
        base=beta_ptr + pid_h,
        shape=(T,),
        strides=(H,),
        offsets=(pid_c * BLOCK_C,),
        block_shape=(BLOCK_C,),
        order=(0,),
    )
    beta = tl.load(beta_blk_ptr, boundary_check=(0,))
    grad_v_blk_ptr = tl.make_block_ptr(
        base=grad_v_ptr + pid_h * D,
        shape=(T, D),
        strides=(H * D, 1),
        offsets=(pid_c * BLOCK_C, 0),
        block_shape=(BLOCK_C, D),
        order=(1, 0),
    )
    grad_v_row = -tl.load(grad_v_blk_ptr, boundary_check=(0,))
    row_dot_blk_ptr = tl.make_block_ptr(
        base=row_dot_ptr + pid_h,
        shape=(T,),
        strides=(H,),
        offsets=(pid_c * BLOCK_C,),
        block_shape=(BLOCK_C,),
        order=(0,),
    )
    row_dot_row = tl.load(row_dot_blk_ptr, boundary_check=(0,)).to(k_ptr.dtype.element_ty)

    for kv_i in range(0, pid_c * BLOCK_C, BLOCK_C):
        k_blk_ptr = tl.make_block_ptr(
            base=k_ptr + pid_h * D,
            shape=(D, T),
            strides=(1, H * D),
            offsets=(0, kv_i),
            block_shape=(D, BLOCK_C),
            order=(0, 1),
        )
        kt = tl.load(k_blk_ptr, boundary_check=(1,))
        qk = tl.dot(k_row, kt) * fa_scale
        p = tl.math.exp2(qk - lse[:, None]) * beta[:, None]

        u_blk_ptr = tl.make_block_ptr(
            base=u_ptr + pid_h * D,
            shape=(D, T),
            strides=(1, H * D),
            offsets=(0, kv_i),
            block_shape=(D, BLOCK_C),
            order=(0, 1),
        )
        ut = tl.load(u_blk_ptr)
        dp = tl.dot(grad_v_row, ut)
        da = p * (dp - row_dot_row[:, None])
        k = tl.trans(kt, 1, 0)
        acc = tl.dot(da.to(k.dtype), k, acc)

    k_row_blk_ptr = tl.make_block_ptr(
        base=k_ptr + pid_h * D,
        shape=(T, D),
        strides=(H * D, 1),
        offsets=(pid_c * BLOCK_C, 0),
        block_shape=(BLOCK_C, D),
        order=(1, 0),
    )
    k_row_true = tl.load(k_row_blk_ptr, boundary_check=(0,))
    qk = tl.dot(k_row, tl.trans(k_row_true, 1, 0)) * fa_scale
    p = tl.math.exp2(qk - lse[:, None]) * beta[:, None]
    u_blk_ptr = tl.make_block_ptr(
        base=u_ptr + pid_h * D,
        shape=(D, T),
        strides=(1, H * D),
        offsets=(0, pid_c * BLOCK_C),
        block_shape=(D, BLOCK_C),
        order=(0, 1),
    )
    ut = tl.load(u_blk_ptr)
    dp = tl.dot(grad_v_row, ut)
    dpm = dp - row_dot_row[:, None]
    mask = block_i[None, :] < block_i[:, None]
    p = tl.where(mask, p, 0.)
    dpm = tl.where(mask, dpm, 0.)
    da = p * dpm
    daat = da
    acc = tl.dot(daat.to(k_row.dtype), k_row_true, acc)

    grad_q_blk_ptr = tl.make_block_ptr(
        base=grad_q_ptr + pid_h * D,
        shape=(T, D),
        strides=(H * D, 1),
        offsets=(BLOCK_C * pid_c, 0),
        block_shape=(BLOCK_C, D),
        order=(1, 0),
    )
    acc = acc * qk_scale
    tl.store(grad_q_blk_ptr, acc.to(grad_q_ptr.dtype.element_ty), boundary_check=(0,))

    daat = tl.trans(da, 1, 0)
    acc = tl.dot(daat.to(k_row.dtype), k_row)
    k_row = k_row_true
    nu = -tl.trans(ut, 1, 0)
    for kv_i in range((pid_c + 1) * BLOCK_C, T, BLOCK_C):
        k_blk_ptr = tl.make_block_ptr(
            base=q_ptr + pid_h * D,
            shape=(D, T),
            strides=(1, H * D),
            offsets=(0, kv_i),
            block_shape=(D, BLOCK_C),
            order=(0, 1),
        )
        kt = tl.load(k_blk_ptr, boundary_check=(1,))
        lse_blk_ptr = tl.make_block_ptr(
            base=lse_ptr + pid_h,
            shape=(T,),
            strides=(H,),
            offsets=(kv_i,),
            block_shape=(BLOCK_C,),
            order=(0,),
        )
        lse = tl.load(lse_blk_ptr, boundary_check=(0,))
        beta_blk_ptr = tl.make_block_ptr(
            base=beta_ptr + pid_h,
            shape=(T,),
            strides=(H,),
            offsets=(kv_i,),
            block_shape=(BLOCK_C,),
            order=(0,),
        )
        beta = tl.load(beta_blk_ptr, boundary_check=(0,))
        qk = tl.dot(k_row, kt) * fa_scale
        p = tl.math.exp2(qk - lse[None, :]) * beta[None, :]

        grad_vt_blk_ptr = tl.make_block_ptr(
            base=grad_v_ptr + pid_h * D,
            shape=(D, T),
            strides=(1, H * D),
            offsets=(0, kv_i),
            block_shape=(D, BLOCK_C),
            order=(0, 1),
        )
        grad_vt = tl.load(grad_vt_blk_ptr, boundary_check=(1,))
        row_dot_blk_ptr = tl.make_block_ptr(
            base=row_dot_ptr + pid_h,
            shape=(T,),
            strides=(H,),
            offsets=(kv_i,),
            block_shape=(BLOCK_C,),
            order=(0,),
        )
        row_dot = tl.load(row_dot_blk_ptr, boundary_check=(0,)).to(k_ptr.dtype.element_ty)
        dp = tl.dot(nu, grad_vt)
        da = p * (dp - row_dot[None, :])
        k = tl.trans(kt, 1, 0)
        acc = tl.dot(da.to(k.dtype), k, acc)

    grad_k_blk_ptr = tl.make_block_ptr(
        base=grad_k_ptr + pid_h * D,
        shape=(T, D),
        strides=(H * D, 1),
        offsets=(BLOCK_C * pid_c, 0),
        block_shape=(BLOCK_C, D),
        order=(1, 0),
    )
    acc = acc * qk_scale
    tl.store(grad_k_blk_ptr, acc.to(grad_k_ptr.dtype.element_ty), boundary_check=(0,))
