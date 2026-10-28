# Copyright 2024 LinkedIn Corporation. BSD-2-Clause; see ../LICENSE.
# Exact function/decorators extracted from the pinned upstream module.
# Loaded in that audited module namespace by opbench.liger; not a standalone import.
@triton.jit
def _grpo_loss_bwd_kernel(
    DLOSS,
    DLOGITS,
    LOGITS,
    OLD_LOGP,
    REF_LOGP,
    INPUT_IDS,
    ADVANTAGES,
    COMPLETION_MASK,
    LSE,
    VLLM_IS_RATIO,
    VLLM_IS_RATIO_STRIDE,
    PHI_SEQ,  # VESPO gamma weight per sequence (B,) or None
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
    loss_stride0,
    loss_stride1,
    L: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr = 4096,
):
    off_b = tl.program_id(0).cast(tl.int64)
    off_l = tl.program_id(1).cast(tl.int64)

    DLOGITS += off_b * (L + 1) * N + off_l * N
    if COMPLETION_MASK is not None:
        COMPLETION_MASK += off_b * L + off_l
        not_skip = tl.load(COMPLETION_MASK)
        if not_skip == 0:
            for start in range(0, N, BLOCK_N):
                cols = tl.arange(0, BLOCK_N) + start
                tl.store(DLOGITS + cols, 0.0, mask=cols < N)
            return

    LOGITS += off_b * (L + 1) * N + off_l * N
    DLOSS += off_b * loss_stride0 + off_l * loss_stride1
    INPUT_IDS += off_b * L + off_l
    ADVANTAGES += off_b
    LSE += off_b * L + off_l

    dloss = tl.load(DLOSS).to(tl.float32)
    lse = tl.load(LSE).to(tl.float32)

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

    # Branch based on loss type for gradient computation
    if LOSS_TYPE == 0:  # GRPO/DAPO/BNPO/DR_GRPO: standard PPO clipping
        coef_2 = tl.clamp(coef_1, 1 - EPS_LOW, 1 + EPS_HIGH)
        if DELTA != 0.0:
            coef_1_for_loss = tl.minimum(coef_1, DELTA)
        else:
            coef_1_for_loss = coef_1
        per_token_loss1 = coef_1_for_loss * advantage
        per_token_loss2 = coef_2 * advantage
        mask = per_token_loss2 >= per_token_loss1
        # Gradient uses original coef_1; zero when delta-clamped (constant → no gradient)
        dlogp = -coef_1 * advantage * mask
        if DELTA != 0.0:
            dlogp = dlogp * (coef_1 <= DELTA)

    elif LOSS_TYPE == 1:  # CISPO: coef_2 is DETACHED, so gradient only flows through logp
        # loss = -coef_2 * advantage * logp, where coef_2 = clamp(coef_1, max=eps_high).detach()
        # d(loss)/d(logp) = -coef_2 * advantage (coef_2 treated as constant due to detach)
        coef_2 = tl.minimum(coef_1, EPS_HIGH)
        dlogp = -coef_2 * advantage

    elif LOSS_TYPE == 2:  # SAPO: gradient through sigmoid gating
        # loss = -sapo_coef * advantage, where sapo_coef = sigmoid(τ*(ρ-1)) * 4/τ
        # d(loss)/d(logp) = -advantage * d(sapo_coef)/d(coef_1) * d(coef_1)/d(logp)
        # d(coef_1)/d(logp) = coef_1 (since coef_1 = exp(logp - old_logp))
        # d(sapo_coef)/d(coef_1) = d/d(coef_1)[sigmoid(τ*(coef_1-1)) * 4/τ]
        #                       = τ * sigmoid' * 4/τ = 4 * sigmoid * (1 - sigmoid)
        # (the τ factors cancel out in the derivative)
        temperature = tl.where(advantage > 0, SAPO_TEMP_POS, SAPO_TEMP_NEG)
        sigmoid_input = temperature * (coef_1 - 1.0)
        sigmoid_val = tl.sigmoid(sigmoid_input)
        d_sapo_d_coef1 = 4.0 * sigmoid_val * (1.0 - sigmoid_val)
        dlogp = -advantage * d_sapo_d_coef1 * coef_1

    elif LOSS_TYPE == 3:  # VESPO: detached gamma weighting on per-token logp
        # loss = -phi_seq * advantage * logp; phi_seq is detached → ∂loss/∂logp = -phi_seq * advantage
        phi_seq = tl.load(PHI_SEQ + off_b).to(tl.float32)
        dlogp = -phi_seq * advantage

    # Apply vLLM IS ratio to PPO gradient (before KL gradient)
    # VESPO folds vllm correction into phi_seq, so we skip it here.
    if VLLM_IS_RATIO is not None and LOSS_TYPE != 3:
        # Use modulo to support both (B, L) per-token and (B, 1) per-sequence shapes
        vllm_is_ratio = tl.load(VLLM_IS_RATIO + off_b * VLLM_IS_RATIO_STRIDE + off_l % VLLM_IS_RATIO_STRIDE).to(
            tl.float32
        )
        dlogp = dlogp * vllm_is_ratio

    if BETA != 0.0:
        REF_LOGP += off_b * L + off_l
        ref_logp = tl.load(REF_LOGP).to(tl.float32)
        if USE_BIAS_CORRECTION_KL:
            # d(kl * coef_1)/d(logp) = coef_1 * (logp - ref_logp), where coef_1 = exp(logp - old_logp)
            dlogp += BETA * coef_1 * (logp - ref_logp)
        else:
            dlogp += BETA * (1 - tl.exp(ref_logp - logp))

    dlogp = dlogp * dloss / TEMPERATURE
    if dlogp == 0.0:
        for start_n in tl.range(0, N, BLOCK_N):
            cols = start_n + tl.arange(0, BLOCK_N)
            tl.store(DLOGITS + cols, 0.0, mask=cols < N)
        return
    tl.debug_barrier()
    for start_n in tl.range(0, N, BLOCK_N):
        cols = start_n + tl.arange(0, BLOCK_N)
        logits = tl.load(LOGITS + cols, mask=cols < N, other=-float("inf")).to(tl.float32) / TEMPERATURE
        probs = tl.exp(logits - lse)
        dlogits = tl.where(cols == idx, 1 - probs, -probs) * dlogp
        tl.store(DLOGITS + cols, dlogits, mask=cols < N)
