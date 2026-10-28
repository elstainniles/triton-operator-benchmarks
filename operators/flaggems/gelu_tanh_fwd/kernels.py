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
from triton.language.extra.cann.libdevice import pow
tanh = tl.math.tanh


@triton.jit
def gelu_tanh(x):
    x_fp32 = x.to(tl.float32)
    output = 0.5 * x * (1 + tanh(x_fp32 * 0.79788456 * (1 + 0.044715 * pow(x_fp32, 2))))
    return output


# Explicit contiguous adapter; not FlagGems pointwise_dynamic.
@triton.jit
def pointwise_probe(X, Y, C, O, N, B: tl.constexpr, KIND: tl.constexpr, FN: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    mask = i < N
    x = tl.load(X + i, mask, other=0)
    if KIND == 0:
        y = tl.load(Y + i, mask, other=0)
        c = tl.load(C + i, mask, other=0)
        z = FN(c, x, y)
    elif KIND == 1:
        z = FN(x)
    else:
        y = tl.load(Y + i, mask, other=0)
        z = FN(x, y)
    tl.store(O + i, z, mask)
