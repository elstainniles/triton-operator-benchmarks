# User-provided original kernel; collection name requested by the contributor.
# Function bodies/signatures/decorators are unchanged. See ../SOURCE_NOTICE.md.
import triton
import triton.language as tl

@triton.jit
def _cross_entropy_fwd_kernel(
    # Outputs
    loss_ptr,
    partial_sum_exp_ptr,
    # Inputs
    logits_ptr,
    labels_ptr,
    row_max_ptr,
    # Params
    ignore_index,
    # Shapes
    n_rows,
    n_cols,
    logits_row_stride,
    # Constexpr
    BLOCK_SIZE: tl.constexpr,
):
    """前向 kernel: 2D grid (n_rows, n_blocks)，每个 block 独立。

    用预计算的 row_max 算 partial sum_exp 和 partial loss。
    """
    row_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)

    logits_ptr = logits_ptr + row_idx * logits_row_stride.to(tl.int64)
    col_offsets = col_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float('inf')).to(tl.float32)

    # partial sum_exp（用 global max，bit-exact with ref）
    row_max = tl.load(row_max_ptr + row_idx)
    block_sum_exp = tl.sum(tl.exp(logits - row_max), axis=0)
    tl.store(partial_sum_exp_ptr + col_block_idx * n_rows + row_idx, block_sum_exp)

    # partial loss: 只在 label 所在 block 输出 -logits_label
    label_idx = tl.load(labels_ptr + row_idx)
    block_start = col_block_idx * BLOCK_SIZE
    block_end = tl.minimum(block_start + BLOCK_SIZE, n_cols)

    if label_idx == ignore_index:
        partial_loss = 0.0
    elif label_idx >= block_start and label_idx < block_end:
        logits_label = tl.load(logits_ptr + label_idx).to(tl.float32)
        partial_loss = row_max - logits_label
    else:
        partial_loss = 0.0

    tl.store(loss_ptr + col_block_idx * n_rows + row_idx, partial_loss)
