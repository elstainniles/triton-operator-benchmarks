# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: fla/ops/rwkv7/channel_mixing.py:206 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
# Source archive: dependencies and launch adaptation are not executable here.
# The complete original file and dependency snapshots are under upstream/.
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': block_size})
        for block_size in [128, 256, 512, 1024, 2048, 4096, 8192]
    ],
    key=['hidden_dim'],
    use_cuda_graph=USE_CUDA_GRAPH,
    **autotune_cache_kwargs,
)
@triton.jit
def rwkv_mix_bwd_kenel(
    dk1_ptr0,
    xk_ptr,
    dx_ptr,
    dx_prev_ptr,
    batch_size,
    token_length,
    hidden_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    batch_idx = offsets // (token_length * hidden_dim)
    seq_feat = offsets % (token_length * hidden_dim)
    seq_idx = seq_feat // hidden_dim
    feat_idx = seq_feat % hidden_dim

    is_valid = offsets < (batch_size * token_length * hidden_dim)

    dk1 = tl.load(dk1_ptr0 + offsets, mask=is_valid)
    xk = tl.load(xk_ptr + feat_idx, mask=is_valid)
    prod = dk1 * xk

    mask_next = seq_idx < (token_length - 1)
    next_offset = offsets + hidden_dim
    dk1_next = tl.load(dk1_ptr0 + next_offset, mask=mask_next & is_valid, other=0.0)
    prod_next = dk1_next * xk
    dx_val = dk1 - prod + tl.where(mask_next, prod_next, 0.0)
    dx_val = tl.cast(dx_val, dtype=dx_ptr.dtype.element_ty, fp_downcast_rounding='rtne')
    tl.store(dx_ptr + offsets, dx_val, mask=is_valid)

    dx_prev_offset = batch_idx * hidden_dim + feat_idx
    is_first_step = seq_idx == 0

    tl.store(
        dx_prev_ptr + dx_prev_offset,
        tl.cast(prod, dtype=dx_prev_ptr.dtype.element_ty),
        mask=is_first_step,
    )
