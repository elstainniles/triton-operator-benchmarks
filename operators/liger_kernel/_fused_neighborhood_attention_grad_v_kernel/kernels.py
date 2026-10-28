# Copyright 2024 LinkedIn Corporation. BSD-2-Clause; see ../LICENSE.
# Exact function/decorators extracted from the pinned upstream module.
# Loaded in that audited module namespace by opbench.liger; not a standalone import.
@triton.jit
def _fused_neighborhood_attention_grad_v_kernel(
    Attn_ptr,
    grad_output_ptr,
    grad_V_ptr,
    attn_batch_stride,
    attn_head_stride,
    attn_seq_stride,
    attn_seq2_stride,
    grad_out_batch_stride,
    grad_out_head_stride,
    grad_out_seq_stride,
    grad_out_dim_stride,
    grad_v_batch_stride,
    grad_v_head_stride,
    grad_v_seq_stride,
    grad_v_dim_stride,
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
    Compute gradient with respect to values: grad_V = Attn^T @ grad_output.

    This kernel computes the gradient of the loss with respect to the value tensor
    by multiplying the transpose of the attention weights with the gradient of the
    output. The computation follows the chain rule for the attention mechanism.

    Args:
        Attn_ptr: Pointer to attention weights [batch_size, num_heads, seq_len, seq_len]
        grad_output_ptr: Pointer to gradient of output [batch_size, num_heads, seq_len, head_dim]
        grad_V_ptr: Pointer to output gradient tensor [batch_size, num_heads, seq_len, head_dim]
        attn_*_stride: Strides for attention weights tensor
        grad_out_*_stride: Strides for gradient output tensor
        grad_v_*_stride: Strides for gradient value tensor
        batch_size: Number of batches
        num_heads: Number of attention heads
        seq_len: Sequence length
        head_dim: Dimension of each attention head
        BLOCK_SIZE_M: Block size for sequence dimension (rows)
        BLOCK_SIZE_N: Block size for head dimension (cols)
        BLOCK_SIZE_K: Block size for sequence dimension (reduction)
        num_stages: Number of pipeline stages
        num_warps: Number of warps

    Grid: (batch_size * num_heads, cdiv(seq_len, BLOCK_SIZE_M), cdiv(head_dim, BLOCK_SIZE_N))
    Each program computes a tile of the value gradient matrix.
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

        attn_ptrs = (
            Attn_ptr
            + batch_id * attn_batch_stride
            + head_id * attn_head_stride
            + k_offsets[:, None] * attn_seq_stride
            + row_offsets[None, :] * attn_seq2_stride
        )
        attn_mask = k_mask[:, None] & (row_offsets[None, :] < seq_len)
        attn_chunk = tl.load(attn_ptrs, mask=attn_mask, other=0.0)

        grad_out_ptrs = (
            grad_output_ptr
            + batch_id * grad_out_batch_stride
            + head_id * grad_out_head_stride
            + k_offsets[:, None] * grad_out_seq_stride
            + col_offsets[None, :] * grad_out_dim_stride
        )
        grad_out_mask = k_mask[:, None] & (col_offsets[None, :] < head_dim)
        grad_out_chunk = tl.load(grad_out_ptrs, mask=grad_out_mask, other=0.0)

        acc += tl.dot(tl.trans(attn_chunk), grad_out_chunk)

    grad_v_ptrs = (
        grad_V_ptr
        + batch_id * grad_v_batch_stride
        + head_id * grad_v_head_stride
        + row_offsets[:, None] * grad_v_seq_stride
        + col_offsets[None, :] * grad_v_dim_stride
    )
    valid_mask = (row_offsets[:, None] < seq_len) & (col_offsets[None, :] < head_dim)
    tl.store(grad_v_ptrs, acc, mask=valid_mask)
