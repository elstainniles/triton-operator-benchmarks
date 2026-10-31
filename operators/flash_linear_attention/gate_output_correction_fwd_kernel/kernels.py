# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: fla/ops/rwkv7/gate_output_correction.py:74 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
# Source archive: dependencies and launch adaptation are not executable here.
# The complete original file and dependency snapshots are under upstream/.
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BT': BT}, num_warps=num_warps)
        for num_warps in [2, 4, 8]
        for BT in [2, 4, 8]
    ],
    key=['num_heads', 'head_dim', 'BLOCK_SIZE_D'],
    **autotune_cache_kwargs,
)
@triton.jit
def gate_output_correction_fwd_kernel(
    o_ptr, r_ptr, k_ptr, r_k_ptr, v_ptr, g_ptr, output_ptr,
    o_b_stride, o_t_stride,
    r_b_stride, r_t_stride, r_h_stride,
    v_b_stride, v_t_stride, v_h_stride,
    r_k_h_stride,
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

    offset_rk_h = pid_h * r_k_h_stride
    vec_r_k = tl.load(r_k_ptr + offset_rk_h + d_idx, mask=mask_d, other=0.0).to(tl.float32)

    offset_rh = pid_b * r_b_stride + t_idx * r_t_stride + pid_h * r_h_stride
    offset_vh = pid_b * v_b_stride + t_idx * v_t_stride + pid_h * v_h_stride
    vec_r = tl.load(r_ptr + offset_rh + d_idx, mask=mask, other=0.0).to(tl.float32)
    vec_k = tl.load(k_ptr + offset_rh + d_idx, mask=mask, other=0.0).to(tl.float32)
    vec_v = tl.load(v_ptr + offset_vh + d_idx, mask=mask, other=0.0).to(tl.float32)
    correction = tl.sum(vec_r * vec_k * vec_r_k, axis=1)[:, None] * vec_v

    offset_o = pid_b * o_b_stride + t_idx * o_t_stride + pid_h * head_dim
    vec_o = tl.load(o_ptr + offset_o + d_idx, mask=mask, other=0.0).to(tl.float32)
    vec_g = tl.load(g_ptr + offset_o + d_idx, mask=mask, other=0.0).to(tl.float32)
    final_output = (vec_o + correction) * vec_g

    tl.store(output_ptr + offset_o + d_idx, final_output.to(output_ptr.dtype.element_ty), mask=mask)
