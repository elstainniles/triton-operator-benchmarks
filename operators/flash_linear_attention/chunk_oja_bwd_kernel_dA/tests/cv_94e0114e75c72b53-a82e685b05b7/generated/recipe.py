"""Existing independent CPU equations; file-generation main omitted."""
import torch
D = 64
T = 512
BT = 64
H = 4
B = 2

def make_fixtures():
    gen = torch.Generator(device='cpu').manual_seed(SEED)
    k = torch.randn(B, T, H, D, generator=gen) * 0.25
    beta = torch.rand(B, T, H, generator=gen)
    chunks = k.reshape(B, T // BT, BT, H, D).permute(0, 1, 3, 2, 4).double()
    b = beta.reshape(B, T // BT, BT, H).permute(0, 1, 3, 2).double()
    result = torch.tril(chunks @ chunks.transpose(-1, -2) * b.unsqueeze(-1), -1).permute(0, 1, 3, 2, 4).contiguous().reshape(B, T, H, BT).float()
    v = torch.randn(B, T, H, D, generator=gen) * 0.25
    gv = torch.randn(B, T, H, D, generator=gen) * 0.1
    do = torch.randn(B, T, H, D, generator=gen) * 0.25
    scale = torch.tensor(0.125, dtype=torch.float32)

    def split(x):
        return x.reshape(B, T // BT, BT, H, D).permute(0, 1, 3, 2, 4).double()
    a, c, gate = (split(do), split(v), split(gv))
    result = a * gate.exp() @ (c * (-gate).exp()).transpose(-1, -2) * scale.item()
    result = torch.tril(result, 0).permute(0, 1, 3, 2, 4).contiguous().reshape(B, T, H, BT).float()
    return {name + '.pt': value.contiguous() if value.ndim else value for name, value in {'v': v, 'gv': gv, 'do': do, 'scale': scale, 'block_size': torch.tensor(BT), 'output': result}.items()}
SEED = 20261030
