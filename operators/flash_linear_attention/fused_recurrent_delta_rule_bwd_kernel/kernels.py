# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: tests/ops/ascend_ops/vv_operator/fused_recurrent_delta_rule_bwd_kernel/test_fused_recurrent_delta_rule_bwd_kernel_with_cpu_golden_npu.py:83 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
# Source archive: dependencies and launch adaptation are not executable here.
# The complete original file and dependency snapshots are under upstream/.
import triton
import triton.language as tl


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_delta_rule_bwd_kernel(
    q,
    k,
    v,
    beta,
    h0,
    dh0,
    dht,
    do,
    dq,
    dk,
    dv,
    db,
    cu_seqlens,
    scale,
    B: tl.constexpr,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NK: tl.constexpr,
    IS_BETA_HEADWISE: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_k, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    mask_k = i_k * BK + tl.arange(0, BK) < K
    mask_v = i_v * BV + tl.arange(0, BV) < V

    p_q = q + (bos * H + i_h) * K + i_k * BK + tl.arange(0, BK) + (T - 1) * H*K
    p_k = k + (bos * H + i_h) * K + i_k * BK + tl.arange(0, BK) + (T - 1) * H*K
    p_v = v + (bos * H + i_h) * V + i_v * BV + tl.arange(0, BV) + (T - 1) * H*V
    p_do = do + (bos * H + i_h) * V + i_v * BV + tl.arange(0, BV) + (T - 1) * H*V
    p_dk = dk + ((i_v * all + bos) * H + i_h) * K + i_k * BK + tl.arange(0, BK) + (T - 1) * H*K
    p_dv = dv + ((i_k * all + bos) * H + i_h) * V + i_v * BV + tl.arange(0, BV) + (T - 1) * H*V
    if IS_BETA_HEADWISE:
        p_beta = beta + (bos + T - 1) * H*V + i_h * V + i_v * BV + tl.arange(0, BV)
        p_dbeta = db + ((i_v * NK + i_k) * all + bos + T - 1) * H*V + i_h * V + tl.arange(0, BV)
    else:
        p_beta = beta + (bos + T - 1) * H + i_h
        p_dbeta = db + (i_v * all + bos + T - 1) * H + i_h

    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_FINAL_STATE_GRADIENT:
        p_ht = dht + i_nh * K * V + (i_k * BK + tl.arange(0, BK)[:, None]) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        b_dh += tl.load(p_ht, mask=mask_k[:, None] & mask_v[None, :], other=0).to(tl.float32)

    for _ in range(T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32) * scale
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask_v, other=0).to(tl.float32)
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)
        else:
            b_beta = tl.load(p_beta).to(tl.float32)
        b_dh += b_q[:, None] * b_do[None, :]
        b_dk = tl.sum(b_dh * (b_v * b_beta)[None, :], axis=1)
        b_dv = tl.sum(b_dh * b_k[:, None], axis=0)

        b_db = b_dv * b_v if IS_BETA_HEADWISE else tl.sum(b_dv * b_v)
        b_dv = b_dv * b_beta

        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=mask_k)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), mask=mask_v)
        if IS_BETA_HEADWISE:
            tl.store(p_dbeta, b_db.to(p_dbeta.dtype.element_ty), mask=mask_v)
        else:
            tl.store(p_dbeta, b_db.to(p_dbeta.dtype.element_ty))

        b_dh -= b_k[:, None] * b_dv[None, :]

        p_q -= H*K
        p_k -= H*K
        p_v -= H*V
        p_do -= H*V
        p_dk -= H*K
        p_dv -= H*V
        p_dbeta -= H * (V if IS_BETA_HEADWISE else 1)
        p_beta -= H * (V if IS_BETA_HEADWISE else 1)

    if USE_INITIAL_STATE:
        p_dh0 = dh0 + i_nh * K * V + (i_k * BK + tl.arange(0, BK)[:, None]) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), mask=mask_k[:, None] & mask_v[None, :])

    tl.debug_barrier()

    b_h = tl.zeros([BK, BV], dtype=tl.float32)

    p_q = q + (bos * H + i_h) * K + i_k * BK + tl.arange(0, BK)
    p_k = k + (bos * H + i_h) * K + i_k * BK + tl.arange(0, BK)
    p_v = v + (bos * H + i_h) * V + i_v * BV + tl.arange(0, BV)
    if IS_BETA_HEADWISE:
        p_beta = beta + (bos * H + i_h) * V + i_v * BV + tl.arange(0, BV)
    else:
        p_beta = beta + bos * H + i_h
    p_do = do + (bos * H + i_h) * V + i_v * BV + tl.arange(0, BV)
    p_dq = dq + ((i_v * all + bos) * H + i_h) * K + i_k * BK + tl.arange(0, BK)
    p_dk = dk + ((i_v * all + bos) * H + i_h) * K + i_k * BK + tl.arange(0, BK)
    p_dv = dv + ((i_k * all + bos) * H + i_h) * V + i_v * BV + tl.arange(0, BV)

    if USE_INITIAL_STATE:
        mask_h = mask_k[:, None] & mask_v[None, :]
        p_h0 = h0 + i_nh * K * V + (i_k * BK + tl.arange(0, BK)[:, None]) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for _ in range(0, T):
        b_dk = tl.load(p_dk, mask=mask_k, other=0).to(tl.float32)
        b_dv = tl.load(p_dv, mask=mask_v, other=0).to(tl.float32)
        b_dk -= tl.sum(b_dv[None, :] * b_h, axis=1)
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=mask_k)

        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask_v, other=0).to(tl.float32)
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)
        else:
            b_beta = tl.load(p_beta).to(tl.float32)
        b_v *= b_beta

        b_h += b_k[:, None] * b_v[None, :]
        b_dq = b_h * b_do[None, :]
        d_q = tl.sum(b_dq, axis=1) * scale
        tl.store(p_dq, d_q.to(p_dq.dtype.element_ty), mask=mask_k)

        p_k += H*K
        p_v += H*V
        p_do += H*V
        p_dq += H*K
        p_dk += H*K
        p_dv += H*V
        p_beta += H * (V if IS_BETA_HEADWISE else 1)
