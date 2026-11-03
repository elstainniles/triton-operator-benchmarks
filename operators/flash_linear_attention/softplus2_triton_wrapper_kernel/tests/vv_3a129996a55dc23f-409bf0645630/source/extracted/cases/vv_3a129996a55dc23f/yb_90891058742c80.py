# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# REVISED FROM
# https://github.com/shawntan/stickbreaking-attention/blob/main/stickbreaking_attention/sb_varlen/softplus.py

import triton

from triton import language as tl

@triton.jit
def softplus2_triton(x):
    return tl.where(x < 15.0, tl.math.log2(1 + tl.math.exp2(x)), x)
