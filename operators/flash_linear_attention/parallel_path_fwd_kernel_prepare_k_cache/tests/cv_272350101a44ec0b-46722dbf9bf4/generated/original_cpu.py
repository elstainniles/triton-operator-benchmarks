"""Existing independent CPU equations; file-generation main omitted."""
import torch

def prepare_k_cache_torch_reference(k, w1, w2, BS):
    """
    Pure PyTorch reference implementation of parallel_path_fwd_kernel_prepare_k_cache.
    
    This function implements the same logic as the Triton kernel but using pure PyTorch
    operations on CPU with float64 precision for maximum accuracy.
    
    Args:
        k: Input tensor of shape [B, T, H, K]
        w1: Weight tensor of shape [B, T, H, K] (normalized vectors)
        w2: Weight tensor of shape [B, T, H, K] (computed from intra_chunk_preprocess)
        BS: Block size (chunk size)
    
    Returns:
        k_new: Output tensor of shape [B, T, H, K]
    """
    B, T, H, K = k.shape
    NT = (T + BS - 1) // BS
    k_new = torch.zeros_like(k)
    for b in range(B):
        for h in range(H):
            for i_t in range(NT):
                start_idx = i_t * BS
                end_idx = min((i_t + 1) * BS, T)
                actual_BT = end_idx - start_idx
                b_k = k[b, start_idx:end_idx, h, :].clone().to(torch.float64)
                for k_block_idx in range(i_t + 1, NT):
                    w_start = k_block_idx * BS
                    w_end = min((k_block_idx + 1) * BS, T)
                    b_w1 = w1[b, w_start:w_end, h, :].to(torch.float64)
                    b_w2 = w2[b, w_start:w_end, h, :].to(torch.float64)
                    b_A = torch.matmul(b_k, b_w2.t())
                    b_k = b_k - torch.matmul(b_A, b_w1)
                k_new[b, start_idx:end_idx, h, :] = b_k.to(k.dtype)
    return k_new
