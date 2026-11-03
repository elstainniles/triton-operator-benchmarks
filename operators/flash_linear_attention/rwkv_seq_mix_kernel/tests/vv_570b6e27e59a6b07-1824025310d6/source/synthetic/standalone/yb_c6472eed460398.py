# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch

import triton

import triton.language as tl

from yb_74f55558e08fc1 import USE_CUDA_GRAPH
from yb_74f55558e08fc1 import autocast_custom_bwd
from yb_74f55558e08fc1 import autocast_custom_fwd
from yb_74f55558e08fc1 import autotune_cache_kwargs
from yb_74f55558e08fc1 import check_pytorch_version
from yb_74f55558e08fc1 import input_guard

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

@triton.jit
def rwkv_channel_mixing_pow_and_relu(
    in_ptr,
    out_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused ReLU and Power operation: x = ReLU(x)^2"""
    xoffset = tl.program_id(0) * BLOCK_SIZE
    xindex = xoffset + tl.arange(0, BLOCK_SIZE)
    x0 = xindex
    x = tl.load(in_ptr + (x0), None)
    x = tl.maximum(x, 0.0).to(tl.float32)
    x = tl.cast(x * x, dtype=out_ptr.dtype.element_ty, fp_downcast_rounding='rtne')
    tl.store(out_ptr + (x0), x, None)

def rwkv_mix_fwd(x, x_prev, x_k):
    has_batch = x.dim() == 3

    if has_batch:
        batch_size, token_length, hidden_dim = x.shape
    else:
        token_length, hidden_dim = x.shape
        batch_size = 1
        x = x.unsqueeze(0)
        x_prev = x_prev.unsqueeze(0)

    token_length = x.shape[1]
    hidden_dim = x.shape[2]
    total_elements = batch_size * token_length * hidden_dim

    output = torch.empty_like(x)

    def grid(meta): return (
        (total_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],  # grid_0
        1,  # grid_1
        1,   # grid_2
    )

    rwkv_seq_mix_kernel[grid](
        x.contiguous(),
        x_prev.contiguous(),
        x_k.squeeze(),
        output,
        batch_size=batch_size,
        token_length=token_length,
        hidden_dim=hidden_dim,
    )
    if not has_batch:
        output = output.squeeze(0)
    return output

def rwkv_relu_and_square_fwd(x: torch.Tensor, inplace: bool = True):
    """
    Triton implementation of RWKV's ReLU and square operation
    Args:
        x: Input tensor
    Returns:
        Tensor after ReLU and square operations
    """
    x = x.contiguous()
    output = x if inplace else torch.empty_like(x)

    def grid(meta): return (
        (output.numel() + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],  # grid_0
        1,  # grid_1
        1,   # grid_2
    )
    rwkv_channel_mixing_pow_and_relu[grid](
        x,
        output,
        BLOCK_SIZE=4096,
    )

    return output

@triton.jit
def relu_square_bwd_kernel(
    out_ptr,
    forward_input_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    """ReLU(x)^2 backward kernel
    grad_input = grad_output * 2 * x if x > 0 else 0
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    x = tl.load(forward_input_ptr + offsets).to(tl.float32)
    grad = tl.load(out_ptr + offsets).to(tl.float32)

    x = tl.maximum(x, 0.0)

    grad_input = grad * 2 * x

    tl.store(out_ptr + offsets, grad_input.to(out_ptr.dtype.element_ty))

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
