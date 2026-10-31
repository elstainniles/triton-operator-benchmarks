# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: fla/modules/fused_kl_div.py:26 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
# Source archive: dependencies and launch adaptation are not executable here.
# The complete original file and dependency snapshots are under upstream/.
import triton
import triton.language as tl


@triton.jit
def kl_div_kernel(
    logits,
    target_logits,
    loss,
    s_logits,
    s_loss,
    reduction: tl.constexpr,
    N: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
):
    # https://github.com/triton-lang/triton/issues/1058
    # If N*V is too large, i_n * stride will overflow out of int32, so we convert to int64
    i_n = tl.program_id(0).to(tl.int64)

    logits += i_n * s_logits
    target_logits += i_n * s_logits

    # m is the max value. use the notation from the paper
    sm = float('-inf')
    tm = float('-inf')
    # d is the sum. use the notation from the paper
    sd, td = 0.0, 0.0

    NV = tl.cdiv(V, BV)
    for iv in range(0, NV):
        o_x = iv * BV + tl.arange(0, BV)
        # for student
        b_sl = tl.load(logits + o_x, mask=o_x < V, other=float('-inf'))
        b_sm = tl.max(b_sl)
        m_new = tl.maximum(sm, b_sm)
        sd = sd * exp(sm - m_new) + tl.sum(exp(b_sl - m_new))
        sm = m_new
        # for teacher
        b_tl = tl.load(target_logits + o_x, mask=o_x < V, other=float('-inf'))
        b_tm = tl.max(b_tl)
        m_new = tl.maximum(tm, b_tm)
        td = td * exp(tm - m_new) + tl.sum(exp(b_tl - m_new))
        tm = m_new

    b_loss = 0.
    # KL(y_true || y) = exp(y_true) * (log(y_true) - log(y))
    for iv in range(0, NV):
        o_x = iv * BV + tl.arange(0, BV)
        b_sl = tl.load(logits + o_x, mask=o_x < V, other=float('-inf'))
        b_tl = tl.load(target_logits + o_x, mask=o_x < V, other=float('-inf'))
        b_sp_log = b_sl - sm - log(sd)
        b_tp_log = b_tl - tm - log(td)
        b_sp = exp(b_sp_log)
        b_tp = exp(b_tp_log)
        b_kl = tl.where(o_x < V, b_tp * (b_tp_log - b_sp_log), 0)
        b_dl = -b_tp + b_sp
        b_loss += tl.sum(b_kl)
        if reduction == 'batchmean':
            b_dl = b_dl / N
        tl.store(logits + o_x, b_dl, mask=o_x < V)

    # Normalize the loss by the number of elements if reduction is 'batchmean'
    if reduction == 'batchmean':
        b_loss = b_loss / N

    tl.store(loss + i_n * s_loss, b_loss)
