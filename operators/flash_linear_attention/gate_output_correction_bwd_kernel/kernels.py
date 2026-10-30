# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: fla/ops/rwkv7/gate_output_correction.py:126 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
# Source archive: dependencies and launch adaptation are not executable here.
# The complete original file and dependency snapshots are under upstream/.
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BT': BT}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [1, 2, 4]
        for BT in [2, 4, 8]
    ],
    key=['num_heads', 'head_dim', 'BLOCK_SIZE_D'],
    **autotune_cache_kwargs,
)
@triton.jit
def gate_output_correction_bwd_kernel(
    grad_output_ptr, o_ptr, r_ptr, k_ptr, r_k_ptr, v_ptr, g_ptr,
    grad_o_ptr, grad_r_ptr, grad_k_ptr, grad_r_k_intermediate_ptr, grad_v_ptr, grad_g_ptr,
    r_b_stride, r_t_stride, r_h_stride, o_b_stride, o_t_stride, r_k_h_stride,
    T,
    T_OFFSET,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BT: tl.constexpr,
):
    pid_b, pid_t_block = tl.program_id(0), tl.program_id(1)
    pid_h = tl.program_id(2)

    t_local = pid_t_block * BT + tl.arange(0, BT)[:, None]
    t_idx = T_OFFSET + t_local
    mask_t = t_local < T

    d_idx = tl.arange(0, BLOCK_SIZE_D)[None, :]
    mask_d = d_idx < head_dim
    mask = mask_t & mask_d

    rkv_offset = pid_b * r_b_stride + t_idx * r_t_stride + pid_h * r_h_stride
    vec_r = tl.load(r_ptr + rkv_offset + d_idx, mask=mask, other=0.0).to(tl.float32)
    vec_k = tl.load(k_ptr + rkv_offset + d_idx, mask=mask, other=0.0).to(tl.float32)
    vec_v = tl.load(v_ptr + rkv_offset + d_idx, mask=mask, other=0.0).to(tl.float32)

    og_offset = pid_b * o_b_stride + t_idx * o_t_stride + pid_h * head_dim
    vec_o = tl.load(o_ptr + og_offset + d_idx, mask=mask, other=0.0).to(tl.float32)
    vec_g = tl.load(g_ptr + og_offset + d_idx, mask=mask, other=0.0).to(tl.float32)
    vec_grad_output = tl.load(grad_output_ptr + og_offset + d_idx, mask=mask, other=0.0).to(tl.float32)

    offset_rk_h = pid_h * r_k_h_stride
    vec_r_k = tl.load(r_k_ptr + offset_rk_h + d_idx, mask=mask_d, other=0.0).to(tl.float32)

    prod_r_k_rk = vec_r * vec_k * vec_r_k
    corr_scalar = tl.sum(prod_r_k_rk, axis=1)
    corr_vec = corr_scalar[:, None] * vec_v
    gated_input = vec_o + corr_vec

    vec_grad_g = vec_grad_output * gated_input
    vec_grad_gate = vec_grad_output * vec_g
    vec_grad_o = vec_grad_gate
    vec_grad_corr = vec_grad_gate
    vec_grad_v = vec_grad_corr * corr_scalar[:, None]
    grad_corr_s = tl.sum(vec_grad_corr * vec_v, axis=1)[:, None]
    vec_grad_r = grad_corr_s * vec_k * vec_r_k
    vec_grad_k = grad_corr_s * vec_r * vec_r_k
    local_grad_rk = grad_corr_s * vec_r * vec_k

    tl.store(grad_o_ptr + og_offset + d_idx, vec_grad_o.to(grad_o_ptr.dtype.element_ty), mask=mask)
    tl.store(grad_g_ptr + og_offset + d_idx, vec_grad_g.to(grad_g_ptr.dtype.element_ty), mask=mask)
    tl.store(grad_r_ptr + rkv_offset + d_idx, vec_grad_r.to(grad_r_ptr.dtype.element_ty), mask=mask)
    tl.store(grad_k_ptr + rkv_offset + d_idx, vec_grad_k.to(grad_k_ptr.dtype.element_ty), mask=mask)
    tl.store(grad_v_ptr + rkv_offset + d_idx, vec_grad_v.to(grad_v_ptr.dtype.element_ty), mask=mask)
    tl.store(grad_r_k_intermediate_ptr + rkv_offset + d_idx,
             local_grad_rk.to(grad_r_k_intermediate_ptr.dtype.element_ty), mask=mask)
