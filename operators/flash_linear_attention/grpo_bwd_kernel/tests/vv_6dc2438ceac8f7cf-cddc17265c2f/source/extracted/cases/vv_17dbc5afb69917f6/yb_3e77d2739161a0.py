# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# modified from https://github.com/mdy666/mdy_triton/blob/e0a856347bd988e05e0152332bba35f1d33c5b1f/others/grpo/grpo_loss.ipynb
# XHS ID: blueeeee

# https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py
import triton

import triton.language as tl

from yb_8459c530c3d6be import exp
from yb_8459c530c3d6be import log

from yb_74f55558e08fc1 import IS_AMD
from yb_74f55558e08fc1 import autotune_cache_kwargs
from yb_74f55558e08fc1 import input_guard
from yb_74f55558e08fc1 import is_A5_soc_version
from yb_74f55558e08fc1 import A5_UB_BLOCK_SIZE_LIMIT

@triton.autotune(
    configs=[
        triton.Config({},  num_warps=NUM_WARPS, num_stages=NUM_STAGES)
        for NUM_WARPS in [32]
        for NUM_STAGES in [4]
    ],
    key=['B', 'N'],
    **autotune_cache_kwargs,
)
@triton.jit
def grpo_bwd_kernel(
    dloss_ptr,
    dlogits_ptr,
    logits_ptr,
    ref_logp_ptr,
    input_ids_ptr,
    advantages_ptr,
    completion_mask_ptr,
    lse_ptr,
    beta,
    B,
    N,
    L,
    start_idx,
    BLOCK_SIZE: tl.constexpr,
):

    row_idx = tl.program_id(0)  # B*L
    off_b = row_idx // L

    N = tl.cast(N, tl.int64)

    dlogits_ptr += N * (row_idx + off_b)
    base_cols = tl.arange(0, BLOCK_SIZE)
    completion_mask_ptr += row_idx
    not_skip = tl.load(completion_mask_ptr).to(tl.int1)

    if not_skip == 1:
        lse_ptr += row_idx
        dloss_ptr += row_idx
        advantages_ptr += off_b
        ref_logp_ptr += row_idx
        logits_ptr += N * (row_idx + off_b)
        input_ids_ptr += row_idx + (off_b+1) * start_idx
        dloss = tl.load(dloss_ptr).to(tl.float32)
        lse = tl.load(lse_ptr).to(tl.float32)
        idx = tl.load(input_ids_ptr)
        x = tl.load(logits_ptr+idx).to(tl.float32)
        advantage = tl.load(advantages_ptr).to(tl.float32)
        ref_logp = tl.load(ref_logp_ptr)
        # Need for in-place grad.
        tl.debug_barrier()
        logp = x - lse

        dlogp = (beta * (-1.0 * exp(ref_logp - logp) + 1)
                 - advantage) * dloss

        for start_n in tl.range(0, N, BLOCK_SIZE):
            cols = start_n + base_cols
            mask = cols < N
            logits = tl.load(logits_ptr+cols, mask=mask, other=-float('inf')).to(tl.float32)
            probs = exp(logits - lse)
            dlogits = tl.where(cols == idx, 1-probs, -probs) * dlogp

            tl.store(dlogits_ptr+cols, dlogits.to(dlogits_ptr.dtype.element_ty), mask=mask)
    else:
        dlogits = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for start_n in tl.range(0, N, BLOCK_SIZE):
            cols = start_n + base_cols
            mask = cols < N

            tl.store(dlogits_ptr+cols, dlogits.to(dlogits_ptr.dtype.element_ty), mask=mask)
