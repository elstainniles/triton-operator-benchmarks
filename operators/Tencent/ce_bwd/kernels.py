# User-provided original kernel; collection name requested by the contributor.
# Function bodies/signatures/decorators are unchanged. See ../SOURCE_NOTICE.md.
import triton
import triton.language as tl

@triton.jit
def _cross_entropy_bwd_kernel(
    # Outputs (inplace)
    dlogits_ptr,
    # Inputs
    logits_ptr,
    logits_max_ptr,
    sum_exp_ptr,
    labels_ptr,
    dloss_ptr,
    # Params
    ignore_index,
    # Shapes
    n_cols,
    logits_row_stride,
    dlogits_row_stride,
    # Constexpr
    BLOCK_SIZE: tl.constexpr,
    STORE_BF16: tl.constexpr = False,
):
    """反向 kernel: 2D grid (n_rows, n_blocks)，每个 block 独立。

    dlogits = (softmax - one_hot) * dloss。
    STORE_BF16=True 时直接写 bf16，省去后续 .to(bf16) 的一次完整 pass。
    """
    row_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)

    logits_ptr = logits_ptr + row_idx * logits_row_stride.to(tl.int64)
    dlogits_ptr = dlogits_ptr + row_idx * dlogits_row_stride.to(tl.int64)
    col_offsets = col_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    label_idx = tl.load(labels_ptr + row_idx)
    if label_idx != ignore_index:
        dloss = tl.load(dloss_ptr + row_idx)
    else:
        dloss = 0.0

    logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float('inf')).to(tl.float32)
    logits_max = tl.load(logits_max_ptr + row_idx)
    sum_exp = tl.load(sum_exp_ptr + row_idx)
    probs = tl.exp(logits - logits_max) / sum_exp

    dlogits = probs * dloss + tl.where(col_offsets == label_idx, -dloss, 0.0)
    if STORE_BF16:
        tl.store(dlogits_ptr + col_offsets, dlogits.to(tl.bfloat16), mask=mask)
    else:
        tl.store(dlogits_ptr + col_offsets, dlogits, mask=mask)
