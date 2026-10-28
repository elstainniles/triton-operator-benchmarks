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
def swiglu_kernel(input_ptr, output_ptr, M, H, stride_in_m, stride_in_h, stride_out_m, stride_out_h, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_H: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    mask = (offs_m[:, None] < M) & (offs_h[None, :] < H)
    input_a_ptr = input_ptr + offs_m[:, None] * stride_in_m + offs_h[None, :] * stride_in_h
    input_b_ptr = input_ptr + offs_m[:, None] * stride_in_m + (offs_h[None, :] + H) * stride_in_h
    output_ptr = output_ptr + offs_m[:, None] * stride_out_m + offs_h[None, :] * stride_out_h
    x_a = tl.load(input_a_ptr, mask=mask, other=0.0).to(tl.float32)
    x_b = tl.load(input_b_ptr, mask=mask, other=0.0).to(tl.float32)
    silu_x_a = x_a * sigmoid(x_a)
    out = silu_x_a * x_b
    tl.store(output_ptr, out.to(x_a.dtype), mask=mask)
