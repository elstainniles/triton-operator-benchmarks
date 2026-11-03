"""Existing independent CPU equations; file-generation main omitted."""
import torch

def make_fixtures():
    torch.manual_seed(42)
    B, T, H, S = (4, 2048, 4, 256)
    g = torch.randn(B, T, H, S, dtype=torch.float16)
    x = g.double().reshape(B, T // 64, 64, H, S)
    i = x.cumsum(2)
    e = i - x
    gi = (i * 1.4427).float().reshape_as(g)
    ge = (e * 1.4427).float().reshape_as(g)
    return {'test_chunk_rwkv6_fwd_cumsum_kernel_cuda.pt': {'input_data': dict(g=g, chunk_size=64, scale=1.4427, B=B, T=T, H=H, S=S), 'gpu_output': dict(gi=gi, ge=ge)}}
