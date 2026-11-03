"""Existing independent CPU equations; file-generation main omitted."""
import torch.nn.functional as F
import torch

def build(kind, src):
    seed = 20261031
    gen = torch.Generator().manual_seed(seed)

    def r(shape, dtype=torch.float32):
        return (torch.randn(shape, generator=gen) * 0.1).to(dtype)
    B, T, H, D, BT = (4, 2048, 8, 64, 16)
    shape = (B, T, H, D)
    fp16 = torch.float16
    return (dict(q=r(shape, fp16), k=F.normalize(r(shape), p=2, dim=-1).to(fp16), v_new=r(shape, fp16), eta=r((B, T, H, 1), fp16) * 0.005, h=r((B, T // BT, H, D, D), fp16), do_tensor=r(shape, fp16), dh=r((B, T // BT, H, D, D), fp16), dhb=r((B, T // BT, H, 1, D), fp16), scale=torch.tensor(D ** (-0.5)), chunk_size=torch.tensor(BT)), seed)
