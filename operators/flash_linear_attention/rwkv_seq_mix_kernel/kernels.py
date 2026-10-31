# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: fla/ops/rwkv7/channel_mixing.py:39 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
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
def rwkv_seq_mix_kernel(
    x_ptr,
    x_prev_ptr,
    mix_k_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    token_length,
    hidden_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    block_idx = block_start + tl.arange(0, BLOCK_SIZE)[:]

    total_seq_dim = token_length * hidden_dim
    batch_idx = block_idx // total_seq_dim
    seq_and_feat = block_idx % total_seq_dim
    seq_idx = seq_and_feat // hidden_dim
    feat_idx = seq_and_feat % hidden_dim

    is_valid = (batch_idx < batch_size) & (seq_idx < token_length)

    x_idx = batch_idx * total_seq_dim + seq_idx * hidden_dim + feat_idx

    curr_x = tl.load(x_ptr + x_idx, mask=is_valid, other=0.0).to(tl.float32)
    k_value = tl.load(mix_k_ptr + feat_idx).to(tl.float32)

    is_first = seq_idx < 1
    prev_state_idx = batch_idx * hidden_dim + feat_idx
    prev_state = tl.load(x_prev_ptr + prev_state_idx,
                         mask=(is_first & is_valid),
                         other=0.0).to(tl.float32)

    prev_x_idx = x_idx - hidden_dim
    prev_x = tl.load(x_ptr + prev_x_idx,
                     mask=(~is_first & is_valid),
                     other=0.0).to(tl.float32)

    prev_value = tl.where(is_first, prev_state, prev_x)
    state_diff = prev_value - curr_x
    mixed = state_diff * k_value
    result = tl.cast(curr_x + mixed, dtype=output_ptr.dtype.element_ty, fp_downcast_rounding='rtne')
    tl.store(output_ptr + x_idx, result, mask=is_valid)
