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


@triton.jit
def mul_broadcast_2d_kernel(a_ptr, b_ptr, out_ptr, n_elements, n_cols, a_s0: tl.constexpr, a_s1: tl.constexpr, b_s0: tl.constexpr, b_s1: tl.constexpr, out_s0: tl.constexpr, out_s1: tl.constexpr, dtype: tl.constexpr, BLOCK_SIZE: tl.constexpr, IS_BOOL: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    row = offsets // n_cols
    col = offsets - row * n_cols
    a = tl.load(a_ptr + row * a_s0 + col * a_s1, mask=mask)
    b = tl.load(b_ptr + row * b_s0 + col * b_s1, mask=mask)
    out = a & b if IS_BOOL else a * b
    tl.store(out_ptr + row * out_s0 + col * out_s1, out, mask=mask)
