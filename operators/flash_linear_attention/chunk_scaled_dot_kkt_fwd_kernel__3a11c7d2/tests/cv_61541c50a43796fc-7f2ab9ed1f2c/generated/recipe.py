"""Existing independent CPU equations; file-generation main omitted."""
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
    return {name + '.pt': value.contiguous() for name, value in {'k': k, 'beta': beta, 'output': y}.items()}
