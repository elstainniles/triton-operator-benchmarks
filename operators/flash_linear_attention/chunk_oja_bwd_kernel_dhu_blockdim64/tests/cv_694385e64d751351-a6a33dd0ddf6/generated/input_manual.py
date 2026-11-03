"""Existing independent CPU equations; file-generation main omitted."""
import torch.nn.functional as F
import torch

def build(kind, src):
    seed = 42
    gen = torch.Generator().manual_seed(seed)

    def r(shape, dtype=torch.float32):
        return (torch.randn(shape, generator=gen) * 0.1).to(dtype)
    if 'oja_dhu'.startswith('oja_') and 'oja_dhu' != 'oja_dhu':
        B, T, H, D = (2, 512, 4, 64)
        BT = 64
        shape = (B, T, H, D)
        d = dict(q=r(shape), k_new=r(shape), h=r((B, T // BT, H, D, D)), gv_cumsum=r(shape), dA=r((B, T, H, BT)), do=r(shape), scale=torch.tensor(D ** (-0.5)))
        return (d, seed)
    B, T, H, D = (2, 512, 8, 64)
    shape = (B, T, H, D)
    d = {k: r(shape) for k in ['q', 'vg', 'w', 'do', 'dk']}
    d.update(gv=F.logsigmoid(torch.randn(shape, generator=gen) * 0.5 - 1), h0=r((B, H, D, D)), dht=r((B, H, D, D)), scale=torch.tensor(D ** (-0.5)), chunk_size=torch.tensor(32))
    return (d, seed)
