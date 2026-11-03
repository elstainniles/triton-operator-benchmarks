# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# modified from https://github.com/mdy666/mdy_triton/blob/e0a856347bd988e05e0152332bba35f1d33c5b1f/others/grpo/grpo_loss.ipynb
# XHS ID: blueeeee

# https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py
import torch

import triton

import triton.language as tl

from yb_8459c530c3d6be import exp
from yb_8459c530c3d6be import log

from yb_74f55558e08fc1 import IS_AMD
from yb_74f55558e08fc1 import autotune_cache_kwargs
from yb_74f55558e08fc1 import input_guard
from yb_74f55558e08fc1 import is_A5_soc_version
from yb_74f55558e08fc1 import A5_UB_BLOCK_SIZE_LIMIT

NUM_WARPS_AUTOTUNE = [4, 8, 16] if IS_AMD else [4, 8, 16, 32]

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': BLOCK_SIZE},  num_warps=NUM_WARPS, num_stages=NUM_STAGES)
        for BLOCK_SIZE in [1024, 2048, 4096, 8192]
        for NUM_WARPS in NUM_WARPS_AUTOTUNE
        for NUM_STAGES in [1, 2, 4]
    ],
    key=['B', 'N'],
    **autotune_cache_kwargs,
)
@triton.jit
def grpo_fwd_kernel(
    logits_ptr,
    ref_logp_ptr,
    input_ids_ptr,
    advantages_ptr,
    completion_mask_ptr,
    loss_ptr,
    lse_ptr,
    beta,
    save_kl: tl.constexpr,
    B,
    M,
    N,
    L,
    start_idx,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)

    off_b = row_idx // L
    N = tl.cast(N, tl.int64)

    loss_ptr += row_idx

    completion_mask_ptr += row_idx
    not_skip = tl.load(completion_mask_ptr).to(tl.int1)
    if not_skip == 1:
        ref_logp_ptr += row_idx
        lse_ptr += row_idx
        advantages_ptr += off_b
        logits_ptr += N * (row_idx + off_b)
        input_ids_ptr += row_idx + (off_b+1) * start_idx
        base_cols = tl.arange(0, BLOCK_SIZE)

        m_i = -float("inf")
        l_i = 0.0
        for start_n in tl.range(0, N, BLOCK_SIZE):
            cols = start_n + base_cols
            mask = cols < N
            logits = tl.load(logits_ptr+cols, mask=mask, other=-float('inf')).to(tl.float32)
            m_ij = tl.max(logits)
            new_m_i = tl.maximum(m_i, m_ij)
            l_i = l_i * exp(m_i - new_m_i) + tl.sum(exp(logits - new_m_i))
            m_i = new_m_i
        lse = log(l_i) + m_i

        idx = tl.load(input_ids_ptr)
        x = tl.load(logits_ptr+idx).to(tl.float32)
        advantage = tl.load(advantages_ptr).to(tl.float32)
        ref_logp = tl.load(ref_logp_ptr)
        logp = x - lse
        diff = ref_logp - logp
        kl = exp(diff) - diff - 1
        loss = kl * beta - advantage

        tl.store(loss_ptr, loss.to(loss_ptr.dtype.element_ty))
        tl.store(lse_ptr, lse.to(lse_ptr.dtype.element_ty))
        if save_kl:
            tl.store(loss_ptr+M, kl.to(loss_ptr.dtype.element_ty))
    else:
        # store 0
        tl.store(loss_ptr, 0.0)
        if save_kl:
            tl.store(loss_ptr+M, 0.0)

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

class GrpoLoss(torch.autograd.Function):

    @input_guard
    @staticmethod
    def forward(ctx, logits, ref_logp, input_ids, advantages, beta, completion_mask, save_kl, inplace=True):
        ctx.input_shape = logits.shape
        B, L_ADD_1, N = ctx.input_shape
        L = L_ADD_1 - 1
        M = B * L
        input_ids_start_index = input_ids.size(1) - L

        if not save_kl:
            loss = torch.empty(B, L, device=logits.device, dtype=torch.float32)
        else:
            loss = torch.empty(B*2, L, device=logits.device, dtype=torch.float32)

        lse = torch.empty(B, L, device=logits.device, dtype=torch.float32)

        if completion_mask is None:
            completion_mask = torch.ones(B, L, device=logits.device, dtype=torch.int32)
        else:
            loss[:B].masked_fill_(completion_mask.logical_not(), 0.0)

        grpo_fwd_kernel[(M,)](
            logits_ptr=logits,
            ref_logp_ptr=ref_logp,
            input_ids_ptr=input_ids,
            advantages_ptr=advantages,
            completion_mask_ptr=completion_mask,
            loss_ptr=loss,
            lse_ptr=lse,
            beta=beta,
            save_kl=save_kl,
            B=B, M=M, N=N, L=L,
            start_idx=input_ids_start_index,
        )
        ctx.beta = beta
        ctx.save_for_backward(lse, logits, input_ids, advantages, completion_mask)
        ctx.ref_logp = ref_logp
        ctx.inplace = inplace
        return loss

    @input_guard
    @staticmethod
    def backward(ctx, dloss):
        # The grad of logits comes from two parts, the reward part and the kl part
        lse, logits, input_ids, advantages, completion_mask = ctx.saved_tensors
        inplace = ctx.inplace
        B, L_ADD_1, N = ctx.input_shape
        L = L_ADD_1 - 1
        M = B * L

        input_ids_start_index = input_ids.size(1) - L

        # B, L_ADD_1, N
        dlogits = logits if inplace else torch.empty_like(logits)
        BLOCK_SIZE_LIMIT = A5_UB_BLOCK_SIZE_LIMIT[logits.dtype] if is_A5_soc_version() else 65536
        BN = min(BLOCK_SIZE_LIMIT, triton.next_power_of_2(N))

        grpo_bwd_kernel[(M,)](
            dloss_ptr=dloss,
            dlogits_ptr=dlogits,
            logits_ptr=logits,
            ref_logp_ptr=ctx.ref_logp,
            input_ids_ptr=input_ids,
            advantages_ptr=advantages,
            completion_mask_ptr=completion_mask,
            lse_ptr=lse,
            beta=ctx.beta,
            B=B, N=N, L=L,
            BLOCK_SIZE=BN,
            start_idx=input_ids_start_index,
        )
        # The last token in the completion is not used in the loss computation
        # and therefore its gradient should be set to 0
        dlogits[:, -1, :].fill_(0.0)
        return dlogits.view(*ctx.input_shape), None, None, None, None, None, None, None

def fused_grpo_loss(logits, ref_logp, input_ids, advantages,
                    beta=0.1, completion_mask=None, save_kl=False, inplace=False) -> torch.Tensor:
    '''
    compute grpo loss, save memory(no addition usage) and fast speed(6X for A800)

    Args:
        logtits: Tensor, [B, L+1, vocab_size], the origin output of model, it's not logits[:, :-1]
        ref_logp: Tensor, [B, L], the origin output of model, it's not ref_logits[:, :-1]
        input_ids: Tensor, [B, K+L], it's prompt_completion_id, it contains the prompt ids and output ids
        advantages: Tensor, [B], the advantages of each prompt
        beta: float, the weight of kl loss
        completion_mask: Tensor, loss mask
        save_kl: bool, if true will save kl

    Retutn:
        loss: Tensor, [B, L], the loss of grpo, it contains the advantage part and kl part

    NOTE: logits(ref_logits) is computed by these steps
        logits_to_keep = completion_ids.size(1)

        def get_per_token_logits(model, input_ids, attention_mask, logits_to_keep):
            # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
            logits = model(
                input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1
            ).logits
            return logits

        logits = get_per_token_logits(model, prompt_completion_ids, attention_mask, logits_to_keep)
    '''
    out = GrpoLoss.apply(logits, ref_logp, input_ids, advantages, beta, completion_mask, save_kl, inplace)
    if not save_kl:
        return out
    else:
        return out.chunk(2, axis=0)
