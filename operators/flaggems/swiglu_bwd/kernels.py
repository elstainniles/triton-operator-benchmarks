# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Extracted from pinned FlagGems; see operator.json and upstream/.
# Function body preserved; external dispatch/autotune decorators removed.
import triton
import triton.language as tl
sigmoid = tl.sigmoid


@triton.jit
def dswiglu_kernel(grad_out_ptr, input_ptr, grad_in_ptr, M, H, stride_grad_out_m, stride_grad_out_h, stride_in_m, stride_in_h, stride_grad_in_m, stride_grad_in_h, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_H: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    mask = (offs_m[:, None] < M) & (offs_h[None, :] < H)
    grad_out_ptr = grad_out_ptr + offs_m[:, None] * stride_grad_out_m + offs_h[None, :] * stride_grad_out_h
    input_a_ptr = input_ptr + offs_m[:, None] * stride_in_m + offs_h[None, :] * stride_in_h
    input_b_ptr = input_ptr + offs_m[:, None] * stride_in_m + (offs_h[None, :] + H) * stride_in_h
    grad_a_ptr = grad_in_ptr + offs_m[:, None] * stride_grad_in_m + offs_h[None, :] * stride_grad_in_h
    grad_b_ptr = grad_in_ptr + offs_m[:, None] * stride_grad_in_m + (offs_h[None, :] + H) * stride_grad_in_h
    grad_out = tl.load(grad_out_ptr, mask=mask, other=0.0).to(tl.float32)
    x_a = tl.load(input_a_ptr, mask=mask, other=0.0).to(tl.float32)
    x_b = tl.load(input_b_ptr, mask=mask, other=0.0).to(tl.float32)
    sig = sigmoid(x_a)
    silu = x_a * sig
    d_silu = sig + x_a * sig * (1 - sig)
    grad_a = grad_out * x_b * d_silu
    grad_b = grad_out * silu
    tl.store(grad_a_ptr, grad_a.to(x_a.dtype), mask=mask)
    tl.store(grad_b_ptr, grad_b.to(x_a.dtype), mask=mask)
