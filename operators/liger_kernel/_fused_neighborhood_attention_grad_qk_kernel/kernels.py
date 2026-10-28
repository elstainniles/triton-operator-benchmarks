# Copyright 2024 LinkedIn Corporation. BSD-2-Clause; see ../LICENSE.
# Exact function/decorators extracted from the pinned upstream module.
# Loaded in that audited module namespace by opbench.liger; not a standalone import.
@triton.jit
def _fused_neighborhood_attention_grad_qk_kernel(
    grad_attn_ptr,
    K_ptr,
    grad_Q_ptr,
    grad_attn_batch_stride,
    grad_attn_head_stride,
    grad_attn_seq_stride,
    grad_attn_seq2_stride,
    k_batch_stride,
    k_head_stride,
    k_seq_stride,
    k_dim_stride,
    grad_q_batch_stride,
    grad_q_head_stride,
    grad_q_seq_stride,
    grad_q_dim_stride,
    batch_size: tl.constexpr,
    num_heads: tl.constexpr,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    num_stages: tl.constexpr,
    num_warps: tl.constexpr,
):
    """
    Compute gradient with respect to queries: grad_Q = grad_attn @ K * scale.

    This kernel computes the gradient of the loss with respect to the query tensor
    by multiplying the gradient of attention weights with the key tensor. The
    computation follows the chain rule for the attention mechanism.

    Args:
        grad_attn_ptr: Pointer to gradient of attention weights [batch_size, num_heads, seq_len, seq_len]
        K_ptr: Pointer to key tensor [batch_size, num_heads, seq_len, head_dim]
        grad_Q_ptr: Pointer to output gradient tensor [batch_size, num_heads, seq_len, head_dim]
        grad_attn_*_stride: Strides for gradient attention tensor
        k_*_stride: Strides for key tensor
        grad_q_*_stride: Strides for gradient query tensor
        batch_size: Number of batches
        num_heads: Number of attention heads
        seq_len: Sequence length
        head_dim: Dimension of each attention head
        scale: Scaling factor applied to attention scores
        BLOCK_SIZE_M: Block size for sequence dimension (rows)
        BLOCK_SIZE_N: Block size for head dimension (cols)
        BLOCK_SIZE_K: Block size for sequence dimension (reduction)
        num_stages: Number of pipeline stages
        num_warps: Number of warps

    Grid: (batch_size * num_heads, cdiv(seq_len, BLOCK_SIZE_M), cdiv(head_dim, BLOCK_SIZE_N))
    Each program computes a tile of the query gradient matrix.
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

    for k_start in range(0, seq_len, BLOCK_SIZE_K):
        k_offsets = k_start + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_offsets < seq_len

        grad_attn_ptrs = (
            grad_attn_ptr
            + batch_id * grad_attn_batch_stride
            + head_id * grad_attn_head_stride
            + row_offsets[:, None] * grad_attn_seq_stride
            + k_offsets[None, :] * grad_attn_seq2_stride
        )
        grad_attn_mask = (row_offsets[:, None] < seq_len) & k_mask[None, :]
        grad_attn_chunk = tl.load(grad_attn_ptrs, mask=grad_attn_mask, other=0.0)

        k_ptrs = (
            K_ptr
            + batch_id * k_batch_stride
            + head_id * k_head_stride
            + k_offsets[:, None] * k_seq_stride
            + col_offsets[None, :] * k_dim_stride
        )
        k_mask_2d = k_mask[:, None] & (col_offsets[None, :] < head_dim)
        k_chunk = tl.load(k_ptrs, mask=k_mask_2d, other=0.0)

        acc += tl.dot(grad_attn_chunk, k_chunk)

    acc = acc * scale

    grad_q_ptrs = (
        grad_Q_ptr
        + batch_id * grad_q_batch_stride
        + head_id * grad_q_head_stride
        + row_offsets[:, None] * grad_q_seq_stride
        + col_offsets[None, :] * grad_q_dim_stride
    )
    valid_mask = (row_offsets[:, None] < seq_len) & (col_offsets[None, :] < head_dim)
    tl.store(grad_q_ptrs, acc, mask=valid_mask)
