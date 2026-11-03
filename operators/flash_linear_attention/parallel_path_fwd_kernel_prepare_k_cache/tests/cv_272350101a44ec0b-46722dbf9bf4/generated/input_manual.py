"""Existing independent CPU equations; file-generation main omitted."""
import torch.nn.functional as F
import torch

def build(cid):
    g = torch.Generator().manual_seed(42)

    def r(shape, dtype=torch.float32):
        return (torch.randn(shape, generator=g) * 0.1).to(dtype)
    B, T, H, HQ, D, BT = (3, 1024, 2, 8, 64, 32)
    low = torch.bfloat16
    s = (B, T, H, D)
    sq = (B, T, HQ, D)
    return dict(k=r(s, low), w1=F.normalize(r(s), dim=-1), w2=r(s), cu_seqlens=None, BS=BT, use_cache=True)
