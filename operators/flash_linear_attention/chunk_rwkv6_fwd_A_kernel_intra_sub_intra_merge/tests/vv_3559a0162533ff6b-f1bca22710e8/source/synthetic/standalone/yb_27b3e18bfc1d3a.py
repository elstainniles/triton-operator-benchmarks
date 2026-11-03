# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import triton

import triton.language as tl

from yb_8459c530c3d6be import exp2

from yb_74f55558e08fc1 import USE_CUDA_GRAPH
from yb_74f55558e08fc1 import autocast_custom_bwd
from yb_74f55558e08fc1 import autocast_custom_fwd
from yb_74f55558e08fc1 import autotune_cache_kwargs
from yb_74f55558e08fc1 import check_shared_mem
from yb_74f55558e08fc1 import input_guard

@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64]
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['BC'],
    use_cuda_graph=USE_CUDA_GRAPH,
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_rwkv6_fwd_A_kernel_intra_sub_inter(
    q,
    k,
    gi,  # cumulative decay inclusive
    ge,  # cumulative decay exclusive
    A,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    i_i, i_j = i_c // NC, i_c % NC
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if i_t * BT + i_i * BC >= T:
        return
    if i_i <= i_j:
        return

    m_i = i_t * BT + i_i * BC + tl.arange(0, BC) < T

    b_A = tl.zeros([BC, BC], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K

        p_q = tl.make_block_ptr(q + (bos*H+i_h)*K, (T, K), (H*K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_gq = tl.make_block_ptr(ge + (bos*H+i_h)*K, (T, K), (H*K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_k = tl.make_block_ptr(k + (bos*H+i_h)*K, (K, T), (1, H*K), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
        p_gk = tl.make_block_ptr(gi + (bos*H+i_h)*K, (K, T), (1, H*K), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
        p_gn = gi + (bos + i_t * BT + i_i * BC - 1) * H*K + i_h * K + o_k

        # [BK,]
        b_gn = tl.load(p_gn, mask=m_k, other=0)
        # [BC, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_gq = tl.where(m_i[:, None] & m_k, tl.load(p_gq, boundary_check=(0, 1)), float('-inf'))
        b_qg = b_q * exp2(b_gq - b_gn[None, :]) * scale
        # [BK, BC]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_gk = tl.load(p_gk, boundary_check=(0, 1))
        b_kg = b_k * exp2(b_gn[:, None] - b_gk)
        # [BC, BC] using tf32 to improve precision here.
        b_A += tl.dot(b_qg, b_kg)

    p_A = tl.make_block_ptr(A + (bos*H + i_h)*BT, (T, BT), (H*BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
    tl.store(p_A, b_A.to(A.dtype.element_ty), boundary_check=(0, 1))

@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8]
    ],
    key=['BK', 'BT'],
    use_cuda_graph=USE_CUDA_GRAPH,
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_rwkv6_fwd_A_kernel_intra_sub_intra(
    q,
    k,
    gi,
    ge,
    u,
    A,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_i, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    i_j = i_i
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if i_t * BT + i_i * BC >= T:
        return

    o_i = tl.arange(0, BC)
    o_k = tl.arange(0, BK)
    m_k = o_k < K
    m_A = (i_t * BT + i_i * BC + tl.arange(0, BC)) < T
    o_A = (bos + i_t * BT + i_i * BC + tl.arange(0, BC)) * H*BT + i_h * BT + i_j * BC
    p_q = tl.make_block_ptr(q + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT + i_i * BC, 0), (BC, BK), (1, 0))
    p_g = tl.make_block_ptr(ge + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT + i_i * BC, 0), (BC, BK), (1, 0))
    p_qj = q + (bos + i_t * BT + i_j * BC) * H*K + i_h * K + o_k
    p_kj = k + (bos + i_t * BT + i_j * BC) * H*K + i_h * K + o_k
    p_gk = gi + (bos + i_t * BT + i_j * BC) * H*K + i_h * K + o_k

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_g = tl.load(p_g, boundary_check=(0, 1))

    p_u = tl.make_block_ptr(u + i_h * K, (K,), (1,), (0,), (BK,), (0,))
    b_u = tl.load(p_u, boundary_check=(0,))
    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        b_qj = tl.load(p_qj, mask=m_k, other=0).to(tl.float32)
        b_kj = tl.load(p_kj, mask=m_k, other=0).to(tl.float32)
        b_gk = tl.load(p_gk, mask=m_k, other=0).to(tl.float32)
        b_A = tl.sum(b_q * b_kj[None, :] * exp2(b_g - b_gk[None, :]), 1)
        b_A = tl.where(o_i > j, b_A * scale, 0.)
        b_A = tl.where(o_i != j, b_A, tl.sum(b_qj * b_kj * b_u * scale))
        tl.store(A + o_A + j, b_A, mask=m_A)
        p_qj += H*K
        p_kj += H*K
        p_gk += H*K

@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=['BC', 'BK'],
    use_cuda_graph=USE_CUDA_GRAPH,
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_rwkv6_fwd_A_kernel_intra_sub_intra_split(
    q,
    k,
    gi,
    ge,
    u,
    A,
    cu_seqlens,
    chunk_indices,
    scale,
    B: tl.constexpr,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_k, i_tc, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    i_t, i_i = i_tc // NC, i_tc % NC
    i_j = i_i
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        all = T
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
        all = B * T

    if i_t * BT + i_i * BC >= T:
        return

    o_i = tl.arange(0, BC)
    o_k = i_k * BK + tl.arange(0, BK)
    m_k = o_k < K
    m_A = (i_t * BT + i_i * BC + tl.arange(0, BC)) < T

    o_A = (i_k * all + bos + i_t * BT + i_i * BC + tl.arange(0, BC)) * H*BC + i_h * BC
    p_q = tl.make_block_ptr(q + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_g = tl.make_block_ptr(ge + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_qj = q + (bos + i_t * BT + i_j * BC) * H*K + i_h * K + o_k
    p_kj = k + (bos + i_t * BT + i_j * BC) * H*K + i_h * K + o_k
    p_gk = gi + (bos + i_t * BT + i_j * BC) * H*K + i_h * K + o_k

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_g = tl.load(p_g, boundary_check=(0, 1))

    p_u = tl.make_block_ptr(u + i_h * K, (K,), (1,), (i_k * BK), (BK,), (0,))
    b_u = tl.load(p_u, boundary_check=(0,))
    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        b_qj = tl.load(p_qj, mask=m_k, other=0).to(tl.float32)
        b_kj = tl.load(p_kj, mask=m_k, other=0).to(tl.float32)
        b_gk = tl.load(p_gk, mask=m_k, other=0).to(tl.float32)
        b_A = tl.sum(b_q * b_kj[None, :] * exp2(b_g - b_gk[None, :]), 1)
        b_A = tl.where(o_i > j, b_A * scale, 0.)
        b_A = tl.where(o_i != j, b_A, tl.sum(b_qj * b_kj * b_u * scale))
        tl.store(A + o_A + j, b_A, mask=m_A)
        p_qj += H*K
        p_kj += H*K
        p_gk += H*K

@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=['BC'],
    use_cuda_graph=USE_CUDA_GRAPH,
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_rwkv6_fwd_A_kernel_intra_sub_intra_merge(
    A,
    A2,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    NK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        all = T
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
        all = B * T

    if i_t * BT + i_c * BC >= T:
        return

    b_A = tl.zeros([BC, BC], dtype=tl.float32)
    for i_k in range(0, NK):
        p_A = tl.make_block_ptr(A + (i_k*all+bos)*H*BC+i_h*BC, (T, BC), (H*BC, 1), (i_t*BT + i_c*BC, 0), (BC, BC), (1, 0))
        b_A += tl.load(p_A, boundary_check=(0, 1))
    p_A2 = tl.make_block_ptr(A2 + (bos*H+i_h)*BT, (T, BT), (H*BT, 1), (i_t * BT + i_c * BC, i_c * BC), (BC, BC), (1, 0))
    tl.store(p_A2, b_A.to(A2.dtype.element_ty), boundary_check=(0, 1))
