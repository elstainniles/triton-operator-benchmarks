# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: tests/ops/ascend_ops/cv_operator/parallel_simple_gla_bwd/test_parallel_simple_gla_bwd_npu.py:223 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
# Source archive: dependencies and launch adaptation are not executable here.
# The complete original file and dependency snapshots are under upstream/.
import triton
import triton.language as tl


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    'NV': lambda args: triton.cdiv(args['V'], args['BV']),
})
@triton.jit(do_not_specialize=['T'])
def parallel_simple_gla_bwd_kernel(
    q,
    k,
    v,
    g,
    do,
    dq,
    dk,
    dv,
    dg,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_G: tl.constexpr,
):
    i_kv, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_k, i_v = i_kv // NV, i_kv % NV
    i_b, i_h = i_bh // H, i_bh % H
    dq += i_v * B * H * T * K
    dk += i_v * B * H * T * K
    dv += i_k * B * H * T * V
    if USE_G:
        dg += i_kv * B * H * T

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    do += (bos * H + i_h) * V
    dq += (bos * H + i_h) * K
    dk += (bos * H + i_h) * K
    dv += (bos * H + i_h) * V
    if USE_G:
        g += bos * H + i_h
        dg += bos * H + i_h

    parallel_simple_gla_bwd_kernel_dq(
        i_t=i_t,
        i_k=i_k,
        i_v=i_v,
        q=q,
        k=k,
        v=v,
        g=g,
        do=do,
        dq=dq,
        dg=dg,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BS=BS,
        BK=BK,
        BV=BV,
        USE_G=USE_G,
    )
    tl.debug_barrier()
    parallel_simple_gla_bwd_kernel_dkv(
        i_t=i_t,
        i_k=i_k,
        i_v=i_v,
        q=q,
        k=k,
        v=v,
        g=g,
        do=do,
        dk=dk,
        dv=dv,
        dg=dg,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BS=BS,
        BK=BK,
        BV=BV,
        USE_G=USE_G,
    )
