# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import os

import triton

import triton.language as tl

import triton.language.extra.libdevice as tldevice

from ofla_74f55558e08fc1 import IS_GATHER_SUPPORTED
from ofla_74f55558e08fc1 import IS_NVIDIA_BLACKWELL

if os.environ.get('FLA_USE_FAST_OPS', '0') == '1':
    @triton.jit
    def exp(x): return tldevice.fast_expf(x.to(tl.float32))
    @triton.jit
    def exp2(x): return tldevice.exp2(x.to(tl.float32))
    @triton.jit
    def log(x): return tldevice.fast_logf(x.to(tl.float32))
    @triton.jit
    def log2(x): return tldevice.fast_log2f(x.to(tl.float32))
    @triton.jit
    def tanh(x): return tldevice.fast_tanhf(x.to(tl.float32))
else:
    @triton.jit
    def exp(x): return tl.exp(x.to(tl.float32))
    @triton.jit
    def exp2(x): return tl.math.exp2(x.to(tl.float32))
    @triton.jit
    def log(x): return tl.log(x.to(tl.float32))
    @triton.jit
    def log2(x): return tl.log2(x.to(tl.float32))
    @triton.jit
    def tanh(x): return tldevice.tanh(x.to(tl.float32))

if not IS_GATHER_SUPPORTED:
    @triton.jit
    def gather(src, index, axis, _builder=None):
        """
        Gather operation that works when tl.gather is not supported.
        This is a fallback implementation that returns None.
        Just to make triton compiler happy.
        """
        return None
else:
    gather = tl.gather
