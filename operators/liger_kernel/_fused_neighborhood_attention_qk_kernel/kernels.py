# Copyright 2024 LinkedIn Corporation. BSD-2-Clause; see ../LICENSE.
# Exact function/decorators extracted from the pinned upstream module.
# Loaded in that audited module namespace by opbench.liger; not a standalone import.
@triton.jit
def _fused_neighborhood_attention_qk_kernel(
    Q_ptr,
    K_ptr,
    QK_ptr,
    mask_ptr,
    q_batch_stride,
    q_head_stride,
    q_seq_stride,
    q_dim_stride,
    k_batch_stride,
    k_head_stride,
    k_seq_stride,
    k_dim_stride,
    qk_batch_stride,
    qk_head_stride,
    qk_seq_stride,
    qk_seq2_stride,
    batch_size: tl.constexpr,
    num_heads: tl.constexpr,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    scale: tl.constexpr,
    kernel_size: tl.constexpr,
    dilation: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    num_stages: tl.constexpr,
    num_warps: tl.constexpr,
):
    """
    Compute Q @ K^T with neighborhood masking and scaling.

    This kernel performs the first stage of neighborhood attention by computing
    the attention scores between queries and keys, applying scaling, and masking
    positions outside the neighborhood window. The result is a matrix of attention
    scores ready for softmax normalization.

    The computation is tiled across sequence dimensions for memory efficiency.
    Each tile computes a block of the attention score matrix by iterating over
    the head dimension and accumulating dot products.

    Args:
        Q_ptr: Pointer to query tensor [batch_size, num_heads, seq_len, head_dim]
        K_ptr: Pointer to key tensor [batch_size, num_heads, seq_len, head_dim]
        QK_ptr: Pointer to output tensor [batch_size, num_heads, seq_len, seq_len]
        mask_ptr: Pointer to neighborhood mask [seq_len, seq_len]
        q_*_stride: Strides for query tensor
        k_*_stride: Strides for key tensor
        qk_*_stride: Strides for output tensor
        batch_size: Number of batches
        num_heads: Number of attention heads
        seq_len: Sequence length
        head_dim: Dimension of each attention head
        scale: Scaling factor for attention scores (typically 1/sqrt(head_dim))
        kernel_size: Size of the neighborhood window
        dilation: Dilation factor for the neighborhood
        BLOCK_SIZE_M: Block size for sequence dimension (rows)
        BLOCK_SIZE_N: Block size for sequence dimension (cols)
        BLOCK_SIZE_K: Block size for head dimension
        num_stages: Number of pipeline stages
        num_warps: Number of warps

    Grid: (batch_size * num_heads, cdiv(seq_len, BLOCK_SIZE_M), cdiv(seq_len, BLOCK_SIZE_N))
    Each program computes a tile of the attention score matrix.
    """
    batch_head_id = tl.program_id(0)
    tile_m = tl.program_id(1)
    tile_n = tl.program_id(2)

    batch_id = batch_head_id // num_heads
    head_id = batch_head_id % num_heads

    row_start = tile_m * BLOCK_SIZE_M
    col_start = tile_n * BLOCK_SIZE_N

    row_offsets = row_start + tl.arange(0, BLOCK_SIZE_M)
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE_N)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_start in range(0, head_dim, BLOCK_SIZE_K):
        k_offsets = k_start + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_offsets < head_dim

        q_ptrs = (
            Q_ptr
            + batch_id * q_batch_stride
            + head_id * q_head_stride
            + row_offsets[:, None] * q_seq_stride
            + k_offsets[None, :] * q_dim_stride
        )
        q_mask = (row_offsets[:, None] < seq_len) & k_mask[None, :]
        q_chunk = tl.load(q_ptrs, mask=q_mask, other=0.0)

        k_ptrs = (
            K_ptr
            + batch_id * k_batch_stride
            + head_id * k_head_stride
            + col_offsets[:, None] * k_seq_stride
            + k_offsets[None, :] * k_dim_stride
        )
        k_mask = (col_offsets[:, None] < seq_len) & k_mask[None, :]
        k_chunk = tl.load(k_ptrs, mask=k_mask, other=0.0)

        acc += tl.dot(q_chunk, tl.trans(k_chunk))

    acc = acc * scale

    mask_ptrs = mask_ptr + row_offsets[:, None] * seq_len + col_offsets[None, :]
    valid_mask = (row_offsets[:, None] < seq_len) & (col_offsets[None, :] < seq_len)
    neighborhood_mask = tl.load(mask_ptrs, mask=valid_mask, other=0.0)

    acc = tl.where(neighborhood_mask > 0.0, acc, float("-inf"))

    qk_ptrs = (
        QK_ptr
        + batch_id * qk_batch_stride
        + head_id * qk_head_stride
        + row_offsets[:, None] * qk_seq_stride
        + col_offsets[None, :] * qk_seq2_stride
    )
    tl.store(qk_ptrs, acc, mask=valid_mask)
