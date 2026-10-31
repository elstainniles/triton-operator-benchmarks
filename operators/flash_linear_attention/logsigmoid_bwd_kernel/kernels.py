# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors


"""Pinned FLA kernel; body/signature/JIT decorator unchanged.
Only autotune is removed; launch fixes an original candidate.
exp/log use the original FLA_USE_FAST_OPS=0 branch.
"""
import triton
import triton.language as tl

@triton.jit
def exp(x): return tl.exp(x.to(tl.float32))

@triton.jit
def log(x): return tl.log(x.to(tl.float32))

@triton.jit(do_not_specialize=['T'])
def logsigmoid_bwd_kernel(
    x,
    dx,
    dy,
    temperature,
    T,
    D: tl.constexpr,
    stride_x_row,
    stride_dx_row,
    stride_dy_row,
    B: tl.constexpr,
):
    i = tl.program_id(0)
    o_i = i * B + tl.arange(0, B)
    m_i = o_i < T
    row = o_i // D
    col = o_i % D
    x_off = row * stride_x_row + col
    dx_off = row * stride_dx_row + col
    dy_off = row * stride_dy_row + col

    b_x = tl.load(x + x_off, mask=m_i, other=0.).to(tl.float32)
    b_dy = tl.load(dy + dy_off, mask=m_i, other=0.).to(tl.float32)
    b_dx = b_dy * ((1. - tl.sigmoid(b_x)) / temperature)
    tl.store(dx + dx_off, b_dx.to(dx.dtype.element_ty), mask=m_i)
