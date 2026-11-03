"""Existing independent CPU equations; file-generation main omitted."""
import torch.nn.functional as F
import torch

def build(kind, src):
    seed = 20261031
    gen = torch.Generator().manual_seed(seed)

    def r(shape, dtype=torch.float32):
        return (torch.randn(shape, generator=gen) * 0.1).to(dtype)
    B, T, H, D, BT = (2, 512, 4, 128, 32)
    shape = (B, T, H, D)
    return (dict(q=r(shape), k=r(shape), v=r(shape), g=F.logsigmoid(r(shape)), h=r((B, T // BT, H, D, D)), do=r(shape), dh=r((B, T // BT, H, D, D)), dA=r((B, T, H, BT)), dg=r(shape), scale=torch.tensor(D ** (-0.5)), chunk_size=torch.tensor(BT)), seed)
