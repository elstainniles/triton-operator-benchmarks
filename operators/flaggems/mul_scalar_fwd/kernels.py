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


@triton.jit(do_not_specialize=['scalar'])
def mul_scalar_kernel(x_ptr, output_ptr, scalar, n_elements, dtype: tl.constexpr, BLOCK_SIZE: tl.constexpr, IS_BOOL: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    out = x & scalar if IS_BOOL else x * scalar
    tl.store(output_ptr + offsets, out, mask=mask)
