# Copyright 2024 LinkedIn Corporation. BSD-2-Clause; see ../LICENSE.
# Exact function/decorators extracted from the pinned upstream module.
# Loaded in that audited module namespace by opbench.liger; not a standalone import.
@triton.jit
def embedding_forward_kernel(
    embeddings_ptr,
    indices_ptr,
    output_ptr,
    n_elements,
    embedding_dim: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    grid_m = tl.cdiv(n_elements, BLOCK_SIZE_M)
    grid_n = tl.cdiv(embedding_dim, BLOCK_SIZE_N)
    total_2d_blocks = grid_m * grid_n

    for block_idx in tl.range(pid, total_2d_blocks, num_progs):
        block_m = block_idx // grid_n
        block_n = block_idx % grid_n

        start_m = block_m * BLOCK_SIZE_M
        start_n = block_n * BLOCK_SIZE_N

        offsets_m = start_m + tl.arange(0, BLOCK_SIZE_M)
        offsets_m = tl.max_contiguous(offsets_m, BLOCK_SIZE_M)
        mask_m = offsets_m < n_elements
        indices = tl.load(indices_ptr + offsets_m, mask=mask_m, other=0)

        offsets_n = start_n + tl.arange(0, BLOCK_SIZE_N)
        offsets_n = tl.max_contiguous(offsets_n, BLOCK_SIZE_N)
        mask_n = offsets_n < embedding_dim
        block_mask = mask_m[:, None] & mask_n[None, :]

        embedding_offsets = indices[:, None] * embedding_dim + offsets_n[None, :]
        embeddings = tl.load(
            embeddings_ptr + embedding_offsets,
            mask=block_mask,
            other=0.0,
        )

        output_offsets = offsets_m[:, None] * embedding_dim + offsets_n[None, :]
        tl.store(output_ptr + output_offsets, embeddings, mask=block_mask)
