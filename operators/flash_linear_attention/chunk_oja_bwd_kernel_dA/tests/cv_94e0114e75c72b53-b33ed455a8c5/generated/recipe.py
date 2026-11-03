"""Existing independent CPU equations; file-generation main omitted."""
import torch.nn.functional as F
import torch

def make_fixtures():
    seed = 20261030
    rng = torch.Generator(device='cpu').manual_seed(seed)
    B, T, H, D, BT = (4, 2048, 8, 64, 64)
    k = torch.randn(B, T, H, D, generator=rng)
    beta = torch.randn(B, T, H, generator=rng).sigmoid()
    kk = k.reshape(B, T // BT, BT, H, D).permute(0, 1, 3, 2, 4).double()
    bb = beta.reshape(B, T // BT, BT, H).permute(0, 1, 3, 2).double()
    y = torch.tril(kk @ kk.transpose(-1, -2) * bb.unsqueeze(-1), -1).permute(0, 1, 3, 2, 4).reshape(B, T, H, BT).float()
    del k, beta, kk, bb, y
    B, T, H, D, BT = (4, 4096, 4, 128, 64)
    v = F.normalize(torch.randn(B, T, H, D, generator=rng), dim=-1)
    gv = F.logsigmoid(torch.rand(B, T, H, D, generator=rng))
    do = torch.randn(B, T, H, D, generator=rng)
    scale = torch.tensor(D ** (-0.5))
    block_size = torch.tensor(BT)

    def split(x):
        return x.reshape(B, T // BT, BT, H, D).permute(0, 1, 3, 2, 4).double()
    vs, gs, ds = map(split, (v, gv, do))
    parts = []
    for offset in (0, 64):
        term = ds[..., offset:offset + 64] * gs[..., offset:offset + 64].exp() @ (vs[..., offset:offset + 64] * (-gs[..., offset:offset + 64]).exp()).transpose(-1, -2)
        term = torch.tril(term * scale.item()).permute(0, 1, 3, 2, 4).reshape(B, T, H, BT).float()
        parts.append(term)
    partials = torch.stack(parts)
    y = partials.sum(0, dtype=torch.float32)
    return {name + '.pt': value.contiguous() for name, value in {'v': v, 'gv': gv, 'do': do, 'scale': scale, 'block_size': block_size, 'output': y, 'output_partials': partials}.items()}
