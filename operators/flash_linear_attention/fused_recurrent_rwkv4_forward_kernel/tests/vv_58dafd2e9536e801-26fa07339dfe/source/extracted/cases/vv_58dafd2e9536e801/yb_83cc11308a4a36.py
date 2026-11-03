# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from typing import Any, cast

import torch

import triton

import triton.language as tl

from torch import Tensor

from yb_8459c530c3d6be import exp

def get_block_size_c(chans: int) -> int:
    if chans < 32:
        return 32
    if chans < 64:
        return 64
    return 128

@triton.jit
def fused_recurrent_rwkv4_forward_kernel(
    # W
    w_ptr,
    w_s_c,
    # U
    u_ptr,
    u_s_c,
    # K
    k_ptr,
    k_s_b,
    k_s_t,
    k_s_c,
    # V
    v_ptr,
    v_s_b,
    v_s_t,
    v_s_c,
    # State
    state_ptr,
    state_s_b,
    state_s_abe,
    state_s_c,
    # WKV
    wkv_ptr,
    wkv_s_b,
    wkv_s_t,
    wkv_s_c,
    # Output state
    state_out_ptr,
    state_out_s_b,
    state_out_s_abe,
    state_out_s_t,
    state_out_s_c,
    # Params
    chans,
    tsz,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Parallelize over the batch dimension.
    b_idx = tl.program_id(0)
    c_idx = tl.program_id(1)

    cs = (c_idx * BLOCK_SIZE_C) + tl.arange(0, BLOCK_SIZE_C)
    cmask = cs < chans

    # Pointers to the batch (and possibly channel) for the input tensors.
    k_ptr = k_ptr + b_idx * k_s_b
    v_ptr = v_ptr + b_idx * v_s_b
    alpha_ptr = state_ptr + b_idx * state_s_b
    beta_ptr = state_ptr + b_idx * state_s_b + state_s_abe
    eps_ptr = state_ptr + b_idx * state_s_b + 2 * state_s_abe

    # Pointers to the batch (and possibly channel) for the output tensors.
    wkv_ptr = wkv_ptr + b_idx * wkv_s_b
    alpha_out_ptr = state_out_ptr + b_idx * state_out_s_b
    beta_out_ptr = state_out_ptr + b_idx * state_out_s_b + state_out_s_abe
    eps_out_ptr = state_out_ptr + b_idx * state_out_s_b + 2 * state_out_s_abe

    # Loads parameters.
    alpha = tl.load(alpha_ptr + cs * state_s_c, mask=cmask).to(tl.float32)
    beta = tl.load(beta_ptr + cs * state_s_c, mask=cmask).to(tl.float32)
    eps = tl.load(eps_ptr + cs * state_s_c, mask=cmask).to(tl.float32)
    w = tl.load(w_ptr + cs * w_s_c, mask=cmask).to(tl.float32)
    u = tl.load(u_ptr + cs * u_s_c, mask=cmask).to(tl.float32)

    for t in range(tsz):
        kt = tl.load(k_ptr + t * k_s_t + cs * k_s_c, mask=cmask).to(tl.float32)
        vt = tl.load(v_ptr + t * v_s_t + cs * v_s_c, mask=cmask).to(tl.float32)

        ukt = u + kt
        tau = tl.maximum(ukt, eps)
        e1a = exp(eps - tau)
        e2a = exp(ukt - tau)
        wkv = (e1a * alpha + e2a * vt) / (e1a * beta + e2a)
        tl.store(wkv_ptr + t * wkv_s_t + cs * wkv_s_c, wkv, mask=cmask)

        w_eps = w + eps
        eps = tl.maximum(w_eps, kt)
        e1b = exp(w_eps - eps)
        e2b = exp(kt - eps)
        alpha = e1b * alpha + e2b * vt
        beta = e1b * beta + e2b
        tl.store(alpha_out_ptr + t * state_out_s_t + cs * state_out_s_c, alpha, mask=cmask)
        tl.store(beta_out_ptr + t * state_out_s_t + cs * state_out_s_c, beta, mask=cmask)
        tl.store(eps_out_ptr + t * state_out_s_t + cs * state_out_s_c, eps, mask=cmask)

def fused_recurrent_rwkv4_forward(
    w: Tensor,
    u: Tensor,
    k: Tensor,
    v: Tensor,
    state: Tensor,
) -> tuple[Tensor, Tensor]:
    (bsz, tsz, chans) = k.shape

    # New tensors to output.
    wkvs = k.new_empty(bsz, tsz, chans)
    state_out = k.new_empty(bsz, 3, tsz, chans)

    # Constants.
    block_size_c = get_block_size_c(chans)

    def grid(meta: dict[str, Any]) -> tuple[int, ...]:
        return (bsz, triton.cdiv(chans, meta["BLOCK_SIZE_C"]))

    fused_recurrent_rwkv4_forward_kernel[grid](
        # W
        w,
        w.stride(0),
        # U
        u,
        u.stride(0),
        # K
        k,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        # V
        v,
        v.stride(0),
        v.stride(1),
        v.stride(2),
        # State
        state,
        state.stride(0),
        state.stride(1),
        state.stride(3),
        # WKV
        wkvs,
        wkvs.stride(0),
        wkvs.stride(1),
        wkvs.stride(2),
        # Output state
        state_out,
        state_out.stride(0),
        state_out.stride(1),
        state_out.stride(2),
        state_out.stride(3),
        # Params
        chans,
        tsz,
        BLOCK_SIZE_C=block_size_c,
    )

    state_out = torch.cat((state, state_out), dim=2)

    return wkvs, state_out
