# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Source: fla/modules/conv/triton/kernels.py:433 at 5752e83ce7e469cfc0a3145bf20d3c902b716d48
# Source archive: dependencies and launch adaptation are not executable here.
# The complete original file and dependency snapshots are under upstream/.
import triton
import triton.language as tl


@triton.heuristics({
    'USE_ACTIVATION': lambda args: args['y'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit
def compute_dh0_kernel(
    dy,
    y,
    weight,
    dh0,
    cu_seqlens,
    stride_dy_n,
    stride_dy_t,
    T,
    D: tl.constexpr,
    W: tl.constexpr,
    BD: tl.constexpr,
    USE_ACTIVATION: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Compute dh0 (gradient w.r.t. initial_state) in a separate kernel.
    This avoids Triton compiler bugs on some architectures (e.g., GB200).

    Grid: (cdiv(D, BD), N)
    """
    i_d, i_n = tl.program_id(0), tl.program_id(1)

    # Get sequence boundaries
    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        seq_len = eos - bos
        # For varlen, dy is [1, total_T, D], offset by bos
        dy_base = dy + bos * stride_dy_t
    else:
        seq_len = T
        # For non-varlen, dy is [B, T, D], offset by i_n * stride_dy_n
        dy_base = dy + tl.cast(i_n, tl.int64) * stride_dy_n

    o_d = i_d * BD + tl.arange(0, BD)
    m_d = o_d < D

    # For each i_w in [1, W), compute dh0[i_n, :, i_w]
    for i_w in tl.static_range(1, W):
        b_dh0 = tl.zeros([BD], dtype=tl.float32)

        # Accumulate contributions from t = 0 to min(i_w, seq_len) - 1
        for t in tl.static_range(0, W - 1):
            if t < i_w:
                w_idx = i_w - 1 - t

                # Load dy[t, :] relative to dy_base
                p_dy = dy_base + t * stride_dy_t + o_d
                m_t = (t < seq_len) & m_d
                b_dy = tl.load(p_dy, mask=m_t, other=0).to(tl.float32)

                if USE_ACTIVATION:
                    if IS_VARLEN:
                        p_y = y + bos * stride_dy_t + t * stride_dy_t + o_d
                    else:
                        p_y = y + tl.cast(i_n, tl.int64) * stride_dy_n + t * stride_dy_t + o_d
                    b_y = tl.load(p_y, mask=m_t, other=0).to(tl.float32)
                    b_ys = tl.sigmoid(b_y)
                    b_dy = b_dy * b_ys * (1 + b_y * (1 - b_ys))

                # Get weight[:, w_idx]
                b_w_col = tl.load(weight + o_d * W + w_idx, mask=m_d, other=0).to(tl.float32)

                # Accumulate
                b_dh0 += tl.where(m_t, b_dy * b_w_col, 0)

        # Store dh0[i_n, :, i_w]
        p_dh0 = dh0 + i_n * D * W + o_d * W + i_w
        tl.store(p_dh0, b_dh0.to(dh0.dtype.element_ty), mask=m_d)
