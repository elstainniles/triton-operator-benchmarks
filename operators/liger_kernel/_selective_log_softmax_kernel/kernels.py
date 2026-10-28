# Copyright 2024 LinkedIn Corporation. BSD-2-Clause; see ../LICENSE.
# Exact function/decorators extracted from the pinned upstream module.
# Loaded in that audited module namespace by opbench.liger; not a standalone import.
@triton.jit
def _selective_log_softmax_kernel(
    LOGITS,
    INPUT_IDS,
    LOG_P,
    LSE,
    MASK,
    TEMPERATURE: tl.constexpr,
    stride_input_ids_b,
    stride_input_ids_l,
    stride_mask_b,
    stride_mask_l,
    L: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr = 4096,
):
    off_b = tl.program_id(0).cast(tl.int64)
    off_l = tl.program_id(1).cast(tl.int64)

    LOGITS += off_b * (L + 1) * N + off_l * N
    INPUT_IDS += off_b * stride_input_ids_b + off_l * stride_input_ids_l
    LOG_P += off_b * L + off_l

    if MASK is not None:
        MASK += off_b * stride_mask_b + off_l * stride_mask_l
        not_skip = tl.load(MASK)
        if not_skip == 0:
            return

    m_i = float("-inf")
    l_i = 0.0
    for start in range(0, N, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        logits = tl.load(LOGITS + cols, mask=cols < N, other=float("-inf")).to(tl.float32) / TEMPERATURE
        new_m_i = tl.maximum(m_i, tl.max(logits))
        alpha = tl.exp(m_i - new_m_i)
        l_i = l_i * alpha + tl.sum(tl.exp(logits - new_m_i))
        m_i = new_m_i
    lse = m_i + tl.log(l_i)

    ids = tl.load(INPUT_IDS)
    x = tl.load(LOGITS + ids).to(tl.float32) / TEMPERATURE
    logp = x - lse
    tl.store(LOG_P, logp)
    if LSE is not None:
        tl.store(LSE + off_b * L + off_l, lse)
