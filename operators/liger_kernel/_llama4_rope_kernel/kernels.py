# Copyright 2024 LinkedIn Corporation. BSD-2-Clause; see ../LICENSE.
# Exact function/decorators extracted from the pinned upstream module.
# Loaded in that audited module namespace by opbench.liger; not a standalone import.
@triton.jit
def _llama4_rope_kernel(
    q_ptr,
    k_ptr,
    freqs_complex_ptr,
    q_row_stride,
    k_row_stride,
    q_head_stride,
    k_head_stride,
    freqs_row_stride,
    seq_len,
    batch_size,
    imag_sign,
    head_dim_half: tl.constexpr,
    n_q_heads: tl.constexpr,
    n_k_heads: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    H100-optimized RoPE kernel with improved parallelization across heads and dimensions.
    Grid: (batch*seq, head)
    """
    # 2D grid
    pid_bs = tl.program_id(0).to(tl.int64)  # over batch*seq
    pid_h = tl.program_id(1)  # over heads

    batch_idx = pid_bs // seq_len
    seq_idx = pid_bs % seq_len

    # Bounds check
    if batch_idx >= batch_size or seq_idx >= seq_len:
        return

    # Base pointers for this (batch, seq) position
    base_offset = batch_idx * seq_len + seq_idx
    q_base = q_ptr + base_offset * q_row_stride
    k_base = k_ptr + base_offset * k_row_stride
    freq_base = seq_idx * freqs_row_stride

    # Tiling over dim/2
    for d_start in tl.static_range(0, head_dim_half, BLOCK_SIZE):
        d_indices = d_start + tl.arange(0, BLOCK_SIZE)
        mask_d = d_indices < head_dim_half

        # Compute offsets for the block
        freq_offsets = d_indices[:, None] * 2 + tl.arange(0, 2)[None, :]
        # Load the block
        freqs_complex = tl.load(freqs_complex_ptr + freq_base + freq_offsets, mask=mask_d[:, None], other=0.0)
        freqs_real, freqs_imag = tl.split(freqs_complex)
        freqs_imag = freqs_imag * imag_sign

        # Process one query head per program in pid_h
        if pid_h < n_q_heads:
            q_head_ptr = q_base + pid_h * q_head_stride
            q_real = tl.load(q_head_ptr + d_indices * 2, mask=mask_d, other=0.0)
            q_imag = tl.load(q_head_ptr + d_indices * 2 + 1, mask=mask_d, other=0.0)

            # Complex multiply with FMAs: (a+ib)*(c+i d) = (a*c - b*d) + i(a*d + b*c)
            new_q_real = tl.math.fma(q_real, freqs_real, -(q_imag * freqs_imag))
            new_q_imag = tl.math.fma(q_real, freqs_imag, q_imag * freqs_real)

            tl.store(q_head_ptr + d_indices * 2, new_q_real, mask=mask_d)
            tl.store(q_head_ptr + d_indices * 2 + 1, new_q_imag, mask=mask_d)

        # Process one key head per program in pid_h
        if pid_h < n_k_heads:
            k_head_ptr = k_base + pid_h * k_head_stride
            k_real = tl.load(k_head_ptr + d_indices * 2, mask=mask_d, other=0.0)
            k_imag = tl.load(k_head_ptr + d_indices * 2 + 1, mask=mask_d, other=0.0)

            new_k_real = tl.math.fma(k_real, freqs_real, -(k_imag * freqs_imag))
            new_k_imag = tl.math.fma(k_real, freqs_imag, k_imag * freqs_real)

            tl.store(k_head_ptr + d_indices * 2, new_k_real, mask=mask_d)
            tl.store(k_head_ptr + d_indices * 2 + 1, new_k_imag, mask=mask_d)
