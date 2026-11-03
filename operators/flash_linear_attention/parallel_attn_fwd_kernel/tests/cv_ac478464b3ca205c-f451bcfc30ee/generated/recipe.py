"""Existing independent CPU equations; file-generation main omitted."""
import math
import torch

def make_fixtures():
    torch.manual_seed(20261030)
    B, T, H, HQ, K, V = (2, 1024, 8, 16, 128, 128)
    q = torch.randn(B, T, HQ, K, dtype=torch.bfloat16)
    k = torch.randn(B, T, H, K, dtype=torch.bfloat16)
    v = torch.randn(B, T, H, V, dtype=torch.bfloat16)
    g = torch.randn(B, T, HQ, dtype=torch.bfloat16)
    gc = g.float().cumsum(1) * 1.4426950216
    scale = K ** (-0.5)
    out = torch.empty(B, T, HQ, V, dtype=torch.bfloat16)
    lse = torch.empty(B, T, HQ, dtype=torch.float32)
    upper = torch.ones(T, T, dtype=torch.bool).triu(1)
    for b in range(B):
        for hq in range(HQ):
            h = hq // (HQ // H)
            scores = q[b, :, hq].double() @ k[b, :, h].double().T * scale
            gd = gc[b, :, hq].double()
            scores += (gd[:, None] - gd[None, :]) * math.log(2.0)
            scores.masked_fill_(upper, float('-inf'))
            out[b, :, hq] = (torch.softmax(scores, dim=1) @ v[b, :, h].double()).to(torch.bfloat16)
            lse[b, :, hq] = (torch.logsumexp(scores, dim=1) / math.log(2.0)).float()
    d = {'input_data': {'q': q, 'k': k, 'v': v, 'g_cumsum': gc, 'sink_bias': None, 'scale': scale, 'window_size': None, 'cu_seqlens': None, 'chunk_indices': None, 'B': B, 'T': T, 'H': H, 'HQ': HQ, 'K': K, 'V': V}, 'gpu_output': {'o': out, 'lse': lse}}
    return {'test_parallel_attn_fwd_kernel_cuda.pt': d}
