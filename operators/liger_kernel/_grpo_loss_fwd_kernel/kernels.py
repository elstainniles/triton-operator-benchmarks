# Copyright 2024 LinkedIn Corporation. BSD-2-Clause; see ../LICENSE.
# Exact function/decorators extracted from the pinned upstream module.
# Loaded in that audited module namespace by opbench.liger; not a standalone import.
@triton.jit
def _grpo_loss_fwd_kernel(
    LOGITS,
    OLD_LOGP,
    REF_LOGP,
    INPUT_IDS,
    COMPLETION_MASK,
    ADVANTAGES,
    VLLM_IS_RATIO,
    VLLM_IS_RATIO_STRIDE,
    PHI_SEQ,  # VESPO gamma weight per sequence (B,) or None
    LOSS,
    LSE,
    KL,
    IS_CLIPPED,
    # constexpr for the fp64 reason in _selective_log_softmax_kernel above.
    TEMPERATURE: tl.constexpr,
    BETA: tl.constexpr,
    EPS_LOW: tl.constexpr,
    EPS_HIGH: tl.constexpr,
    LOSS_TYPE: tl.constexpr,
    SAPO_TEMP_POS: tl.constexpr,
    SAPO_TEMP_NEG: tl.constexpr,
    DELTA: tl.constexpr,
    USE_BIAS_CORRECTION_KL: tl.constexpr,
    L: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr = 4096,
):
    off_b = tl.program_id(0).cast(tl.int64)
    off_l = tl.program_id(1).cast(tl.int64)

    if COMPLETION_MASK is not None:
        COMPLETION_MASK += off_b * L + off_l
        not_skip = tl.load(COMPLETION_MASK)
        if not_skip == 0:
            return

    LOGITS += off_b * (L + 1) * N + off_l * N
    INPUT_IDS += off_b * L + off_l
    ADVANTAGES += off_b
    LOSS += off_b * L + off_l
    LSE += off_b * L + off_l
    IS_CLIPPED += off_b * L + off_l

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

    idx = tl.load(INPUT_IDS)
    x = tl.load(LOGITS + idx).to(tl.float32) / TEMPERATURE
    logp = x - lse
    if OLD_LOGP is None:
        old_logp = logp
    else:
        OLD_LOGP += off_b * L + off_l
        old_logp = tl.load(OLD_LOGP).to(tl.float32)
    coef_1 = tl.exp(logp - old_logp)
    advantage = tl.load(ADVANTAGES).to(tl.float32)

    # Branch based on loss type
    if LOSS_TYPE == 0:  # GRPO/DAPO/BNPO/DR_GRPO: standard PPO clipping
        coef_2 = tl.clamp(coef_1, 1 - EPS_LOW, 1 + EPS_HIGH)
        is_low_clipped = (coef_1 < 1 - EPS_LOW) & (advantage < 0)
        is_high_clipped = (coef_1 > 1 + EPS_HIGH) & (advantage > 0)
        is_clipped = is_low_clipped | is_high_clipped
        # Apply delta (two-sided clipping from INTELLECT-2) to coef_1
        if DELTA != 0.0:
            coef_1 = tl.minimum(coef_1, DELTA)
        per_token_loss1 = coef_1 * advantage
        per_token_loss2 = coef_2 * advantage
        per_token_loss = -tl.minimum(per_token_loss1, per_token_loss2)

    elif LOSS_TYPE == 1:  # CISPO: upper-bound only clipping, detached, multiply by logp
        # Reference: MiniMax-M1 technical report
        # https://github.com/huggingface/trl/blob/035c3ff151b953ca72cdfe0ee966bc1469a26fde/trl/trainer/grpo_trainer.py#L2030
        coef_2 = tl.minimum(coef_1, EPS_HIGH)  # upper-bound only (EPS_HIGH is the raw bound for CISPO)
        per_token_loss = -coef_2 * advantage * logp  # includes logp term
        is_clipped = (coef_1 > EPS_HIGH) & (advantage > 0)

    elif LOSS_TYPE == 2:  # SAPO: soft adaptive policy optimization with sigmoid gating
        # Reference: https://huggingface.co/papers/2511.20347
        # Formula: sigmoid(τ * (ρ - 1)) * 4 / τ
        temperature = tl.where(advantage > 0, SAPO_TEMP_POS, SAPO_TEMP_NEG)
        sigmoid_input = temperature * (coef_1 - 1.0)
        sapo_coef = tl.sigmoid(sigmoid_input) * 4.0 / temperature
        per_token_loss = -sapo_coef * advantage
        is_clipped = 0.0  # SAPO has no clipping concept

    elif LOSS_TYPE == 3:  # VESPO: detached gamma weighting phi(w) on per-token logp
        # Reference: TRL grpo_trainer.get_gamma_weights / chunked_loss/grpo_loss.py
        # phi_seq is precomputed per-sequence, vllm correction is folded into phi_seq.
        phi_seq = tl.load(PHI_SEQ + off_b).to(tl.float32)
        per_token_loss = -phi_seq * advantage * logp
        is_clipped = 0.0  # VESPO has no clipping concept

    # Apply vLLM importance sampling correction BEFORE adding KL penalty
    # VESPO folds this into phi_seq (in log space), so we skip it here.
    if VLLM_IS_RATIO is not None and LOSS_TYPE != 3:
        # Use modulo to support both (B, L) per-token and (B, 1) per-sequence shapes
        vllm_is_ratio = tl.load(VLLM_IS_RATIO + off_b * VLLM_IS_RATIO_STRIDE + off_l % VLLM_IS_RATIO_STRIDE).to(
            tl.float32
        )
        per_token_loss = per_token_loss * vllm_is_ratio

    if BETA != 0.0:
        REF_LOGP += off_b * L + off_l
        KL += off_b * L + off_l
        ref_logp = tl.load(REF_LOGP).to(tl.float32)
        kl = tl.exp(ref_logp - logp) - (ref_logp - logp) - 1
        if USE_BIAS_CORRECTION_KL:
            # Importance-sampling-corrected KL (DeepSeek-V3.2): kl *= coef_1
            kl = kl * tl.exp(logp - old_logp)
        per_token_loss += BETA * kl
        tl.store(KL, kl)

    tl.store(LOSS, per_token_loss)
    tl.store(LSE, lse)
    tl.store(IS_CLIPPED, is_clipped)
