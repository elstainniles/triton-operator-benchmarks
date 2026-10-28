# Copyright 2024 LinkedIn Corporation. BSD-2-Clause; see ../LICENSE.
# Exact function/decorators extracted from the pinned upstream module.
# Loaded in that audited module namespace by opbench.liger; not a standalone import.
@triton.jit
def _fused_neighborhood_attention_grad_k_kernel(
    grad_attn_ptr,
    Q_ptr,
    grad_K_ptr,
    grad_attn_batch_stride,
    grad_attn_head_stride,
    grad_attn_seq_stride,
    grad_attn_seq2_stride,
    q_batch_stride,
    q_head_stride,
    q_seq_stride,
    q_dim_stride,
    grad_k_batch_stride,
    grad_k_head_stride,
    grad_k_seq_stride,
    grad_k_dim_stride,
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
    Compute gradient with respect to keys: grad_K = grad_attn^T @ Q * scale.

    This kernel computes the gradient of the loss with respect to the key tensor
    by multiplying the transpose of the gradient of attention weights with the
    query tensor. The computation follows the chain rule for the attention mechanism.

    Args:
        grad_attn_ptr: Pointer to gradient of attention weights [batch_size, num_heads, seq_len, seq_len]
        Q_ptr: Pointer to query tensor [batch_size, num_heads, seq_len, head_dim]
        grad_K_ptr: Pointer to output gradient tensor [batch_size, num_heads, seq_len, head_dim]
        grad_attn_*_stride: Strides for gradient attention tensor
        q_*_stride: Strides for query tensor
        grad_k_*_stride: Strides for gradient key tensor
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
    Each program computes a tile of the key gradient matrix.
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

        q_ptrs = (
            Q_ptr
            + batch_id * q_batch_stride
            + head_id * q_head_stride
            + k_offsets[:, None] * q_seq_stride
            + col_offsets[None, :] * q_dim_stride
        )
        q_mask = k_mask[:, None] & (col_offsets[None, :] < head_dim)
        q_chunk = tl.load(q_ptrs, mask=q_mask, other=0.0)

        grad_attn_T_ptrs = (
            grad_attn_ptr
            + batch_id * grad_attn_batch_stride
            + head_id * grad_attn_head_stride
            + row_offsets[:, None] * grad_attn_seq2_stride
            + k_offsets[None, :] * grad_attn_seq_stride
        )
        grad_attn_T_mask = (row_offsets[:, None] < seq_len) & k_mask[None, :]
        grad_attn_T_chunk = tl.load(grad_attn_T_ptrs, mask=grad_attn_T_mask, other=0.0)

        acc += tl.dot(grad_attn_T_chunk, q_chunk)

    acc = acc * scale

    grad_k_ptrs = (
        grad_K_ptr
        + batch_id * grad_k_batch_stride
        + head_id * grad_k_head_stride
        + row_offsets[:, None] * grad_k_seq_stride
        + col_offsets[None, :] * grad_k_dim_stride
    )
    valid_mask = (row_offsets[:, None] < seq_len) & (col_offsets[None, :] < head_dim)
    tl.store(grad_k_ptrs, acc, mask=valid_mask)
