# Copyright 2024 LinkedIn Corporation. BSD-2-Clause; see ../LICENSE.
# Exact function/decorators extracted from the pinned upstream module.
# Loaded in that audited module namespace by opbench.liger; not a standalone import.
@triton.jit
def _fused_neighborhood_attention_grad_attn_kernel(
    grad_output_ptr,
    V_ptr,
    grad_attn_ptr,
    grad_out_batch_stride,
    grad_out_head_stride,
    grad_out_seq_stride,
    grad_out_dim_stride,
    v_batch_stride,
    v_head_stride,
    v_seq_stride,
    v_dim_stride,
    grad_attn_batch_stride,
    grad_attn_head_stride,
    grad_attn_seq_stride,
    grad_attn_seq2_stride,
    batch_size: tl.constexpr,
    num_heads: tl.constexpr,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    num_stages: tl.constexpr,
    num_warps: tl.constexpr,
):
    """
    Compute gradient with respect to attention weights: grad_attn = grad_output @ V^T.

    This kernel computes the gradient of the loss with respect to the attention
    weights by multiplying the gradient of the output with the transpose of the
    value tensor. This gradient will later be passed through the softmax backward
    pass to compute gradients for the attention scores.

    Args:
        grad_output_ptr: Pointer to gradient of output [batch_size, num_heads, seq_len, head_dim]
        V_ptr: Pointer to value tensor [batch_size, num_heads, seq_len, head_dim]
        grad_attn_ptr: Pointer to output gradient tensor [batch_size, num_heads, seq_len, seq_len]
        grad_out_*_stride: Strides for gradient output tensor
        v_*_stride: Strides for value tensor
        grad_attn_*_stride: Strides for gradient attention tensor
        batch_size: Number of batches
        num_heads: Number of attention heads
        seq_len: Sequence length
        head_dim: Dimension of each attention head
        BLOCK_SIZE_M: Block size for sequence dimension (rows)
        BLOCK_SIZE_N: Block size for sequence dimension (cols)
        BLOCK_SIZE_K: Block size for head dimension (reduction)
        num_stages: Number of pipeline stages
        num_warps: Number of warps

    Grid: (batch_size * num_heads, cdiv(seq_len, BLOCK_SIZE_M), cdiv(seq_len, BLOCK_SIZE_N))
    Each program computes a tile of the attention gradient matrix.
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

        grad_out_ptrs = (
            grad_output_ptr
            + batch_id * grad_out_batch_stride
            + head_id * grad_out_head_stride
            + row_offsets[:, None] * grad_out_seq_stride
            + k_offsets[None, :] * grad_out_dim_stride
        )
        grad_out_mask = (row_offsets[:, None] < seq_len) & k_mask[None, :]
        grad_out_chunk = tl.load(grad_out_ptrs, mask=grad_out_mask, other=0.0)

        v_ptrs = (
            V_ptr
            + batch_id * v_batch_stride
            + head_id * v_head_stride
            + col_offsets[None, :] * v_seq_stride
            + k_offsets[:, None] * v_dim_stride
        )
        v_mask = (col_offsets[None, :] < seq_len) & k_mask[:, None]
        v_chunk = tl.load(v_ptrs, mask=v_mask, other=0.0)

        acc += tl.dot(grad_out_chunk, v_chunk)

    grad_attn_ptrs = (
        grad_attn_ptr
        + batch_id * grad_attn_batch_stride
        + head_id * grad_attn_head_stride
        + row_offsets[:, None] * grad_attn_seq_stride
        + col_offsets[None, :] * grad_attn_seq2_stride
    )
    valid_mask = (row_offsets[:, None] < seq_len) & (col_offsets[None, :] < seq_len)
    tl.store(grad_attn_ptrs, acc, mask=valid_mask)
