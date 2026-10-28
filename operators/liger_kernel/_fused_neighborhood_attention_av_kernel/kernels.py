# Copyright 2024 LinkedIn Corporation. BSD-2-Clause; see ../LICENSE.
# Exact function/decorators extracted from the pinned upstream module.
# Loaded in that audited module namespace by opbench.liger; not a standalone import.
@triton.jit
def _fused_neighborhood_attention_av_kernel(
    Attn_ptr,
    V_ptr,
    Out_ptr,
    attn_batch_stride,
    attn_head_stride,
    attn_seq_stride,
    attn_seq2_stride,
    v_batch_stride,
    v_head_stride,
    v_seq_stride,
    v_dim_stride,
    out_batch_stride,
    out_head_stride,
    out_seq_stride,
    out_dim_stride,
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
    Compute Attention @ V to produce the final output.

    This kernel performs the second stage of neighborhood attention by multiplying
    the normalized attention weights with the value matrix. The computation is
    tiled for memory efficiency, with each tile computing a block of the output.

    Args:
        Attn_ptr: Pointer to attention weights [batch_size, num_heads, seq_len, seq_len]
        V_ptr: Pointer to value tensor [batch_size, num_heads, seq_len, head_dim]
        Out_ptr: Pointer to output tensor [batch_size, num_heads, seq_len, head_dim]
        attn_*_stride: Strides for attention weights tensor
        v_*_stride: Strides for value tensor
        out_*_stride: Strides for output tensor
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
    Each program computes a tile of the output matrix.
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
            + row_offsets[:, None] * attn_seq_stride
            + k_offsets[None, :] * attn_seq2_stride
        )
        attn_mask = (row_offsets[:, None] < seq_len) & k_mask[None, :]
        attn_chunk = tl.load(attn_ptrs, mask=attn_mask, other=0.0)

        v_ptrs = (
            V_ptr
            + batch_id * v_batch_stride
            + head_id * v_head_stride
            + k_offsets[:, None] * v_seq_stride
            + col_offsets[None, :] * v_dim_stride
        )
        v_mask = k_mask[:, None] & (col_offsets[None, :] < head_dim)
        v_chunk = tl.load(v_ptrs, mask=v_mask, other=0.0)

        acc += tl.dot(attn_chunk, v_chunk)

    out_ptrs = (
        Out_ptr
        + batch_id * out_batch_stride
        + head_id * out_head_stride
        + row_offsets[:, None] * out_seq_stride
        + col_offsets[None, :] * out_dim_stride
    )
    valid_mask = (row_offsets[:, None] < seq_len) & (col_offsets[None, :] < head_dim)
    tl.store(out_ptrs, acc, mask=valid_mask)
