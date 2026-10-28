# Copyright 2024 LinkedIn Corporation. BSD-2-Clause; see ../LICENSE.
# Exact function/decorators extracted from the pinned upstream module.
# Loaded in that audited module namespace by opbench.liger; not a standalone import.
@triton.jit
def _neighborhood_mask_kernel(
    mask_ptr,
    seq_len: tl.constexpr,
    kernel_size: tl.constexpr,
    dilation: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
    num_warps: tl.constexpr,
):
    """
    Generate a neighborhood attention mask for a given sequence.

    This kernel creates a binary mask that defines which positions in a sequence
    can attend to each other based on a neighborhood window with optional dilation.
    Each row of the mask corresponds to a query position, and each column indicates
    whether that key position is within the allowed neighborhood.

    The neighborhood is defined as positions within kernel_size//2 * dilation distance
    from the center position. When dilation > 1, only positions at multiples of the
    dilation factor are included in the neighborhood.

    Args:
        mask_ptr: Pointer to the output mask tensor [seq_len, seq_len]
        seq_len: Length of the input sequence
        kernel_size: Size of the neighborhood window (must be odd)
        dilation: Dilation factor for the neighborhood pattern
        BLOCK_SIZE: Block size for processing (compile-time constant)
        num_stages: Number of pipeline stages (compile-time constant)
        num_warps: Number of warps (compile-time constant)

    Grid: (seq_len,)
    Each program processes one row of the mask matrix.
    """
    row_id = tl.program_id(0)

    center = row_id
    half_kernel = kernel_size // 2

    start = tl.maximum(0, center - half_kernel * dilation)
    end = tl.minimum(seq_len, center + half_kernel * dilation + 1)

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < seq_len

    valid_neighbors = (col_offsets >= start) & (col_offsets < end)
    if dilation > 1:
        relative_pos = col_offsets - center
        valid_dilation = (relative_pos % dilation) == 0
        valid_neighbors = valid_neighbors & valid_dilation

    mask_values = tl.where(valid_neighbors & mask, 1.0, 0.0)

    base_offset = row_id * seq_len
    tl.store(mask_ptr + base_offset + col_offsets, mask_values, mask=mask)
