# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Fixed CV launch adapter; JIT signatures and bodies match upstream/.
# Autotune/heuristics are bound explicitly by cases-cv.json.
# Generated from exact archived source; function bodies unchanged.
import triton
import triton.language as tl
import math

@triton.jit
def tanh(x):
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x))

@triton.jit
def exp(x):
    return tl.exp(x.to(tl.float32))

@triton.jit
def sigmoid(x):
    return 1.0 / (1.0 + exp(-x))

@triton.jit
def relu(x):
    return tl.maximum(x, 0.0)

@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)

@triton.jit
def matmul_kernel(a, b, c, input, alpha, beta, M, N, K, stride_ab, stride_am, stride_ak, stride_bk, stride_bn, stride_cb, stride_cm, stride_cn, BM: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr, G: tl.constexpr, ACTIVATION: tl.constexpr, HAS_INPUT: tl.constexpr, HAS_ALPHA: tl.constexpr, HAS_BETA: tl.constexpr, ALLOW_TF32: tl.constexpr, X_DIM: tl.constexpr=1):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    i_b, i_m, i_n = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    NM, NN = (tl.num_programs(1), tl.num_programs(2))
    i_m, i_n = tl.swizzle2d(i_m, i_n, NM, NN, G)
    a_batch_ptr = a + i_b * stride_ab
    o_am = (i_m * BM + tl.arange(0, BM)) % M
    o_bn = (i_n * BN + tl.arange(0, BN)) % N
    o_k = tl.arange(0, BK)
    p_a = a_batch_ptr + (o_am[:, None] * stride_am + o_k[None, :] * stride_ak)
    p_b = b + (o_k[:, None] * stride_bk + o_bn[None, :] * stride_bn)
    b_acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        b_a = tl.load(p_a, mask=o_k[None, :] < K - k * BK, other=0.0)
        b_b = tl.load(p_b, mask=o_k[:, None] < K - k * BK, other=0.0)
        b_acc = tl.dot(b_a, b_b, acc=b_acc, allow_tf32=ALLOW_TF32)
        p_a += BK * stride_ak
        p_b += BK * stride_bk
    o_cm = i_m * BM + tl.arange(0, BM)
    o_cn = i_n * BN + tl.arange(0, BN)
    mask = (o_cm[:, None] < M) & (o_cn[None, :] < N)
    b_c = b_acc
    if ACTIVATION == 'leaky_relu':
        b_c = leaky_relu(b_c)
    elif ACTIVATION == 'relu':
        b_c = relu(b_c)
    elif ACTIVATION == 'sigmoid':
        b_c = sigmoid(b_c)
    elif ACTIVATION == 'tanh':
        b_c = tanh(b_c)
    if HAS_ALPHA:
        b_c *= tl.load(alpha)
    if HAS_INPUT:
        p_i = input + (stride_cm * o_cm[:, None] if X_DIM == 2 else 0) + stride_cn * o_cn[None, :]
        mask_p = o_cn[None, :] < N if X_DIM == 1 else mask
        b_i = tl.load(p_i, mask=mask_p, other=0.0).to(tl.float32)
        if HAS_BETA:
            b_i *= tl.load(beta)
        b_c += b_i
    c_batch_ptr = c + i_b * stride_cb
    p_c = c_batch_ptr + stride_cm * o_cm[:, None] + stride_cn * o_cn[None, :]
    tl.store(p_c, b_c.to(c.dtype.element_ty), mask=mask)
