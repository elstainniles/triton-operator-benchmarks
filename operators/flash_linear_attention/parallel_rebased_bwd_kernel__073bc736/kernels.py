# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: tests/ops/ascend_ops/cv_operator/_parallel_rebased_bwd_dkv/test__parallel_rebased_bwd_dkv_npu.py:90 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
# Source archive: dependencies and launch adaptation are not executable here.
# The complete original file and dependency snapshots are under upstream/.
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['T'])
def parallel_rebased_bwd_kernel(
    q,
    k,
    v,
    do,
    dz,
    dq,
    dk,
    dv,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BTL: tl.constexpr,
    BTS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_kv, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    NV = tl.cdiv(V, BV)
    i_k = i_kv // (NV)
    i_v = i_kv % (NV)
    i_h = i_bh % H
    tl.debug_barrier()
    _parallel_rebased_bwd_dkv(
        i_bh,
        i_c,
        i_k,
        i_v,
        i_h,
        q,
        k,
        v,
        do,
        dz,
        dk,
        dv,
        scale,
        B=B,
        H=H,
        T=T,
        K=K,
        V=V,
        BTL=BTL,
        BTS=BTS,
        BK=BK,
        BV=BV,
    )
