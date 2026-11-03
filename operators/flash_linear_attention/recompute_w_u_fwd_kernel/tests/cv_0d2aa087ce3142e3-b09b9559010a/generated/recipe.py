"""Existing independent CPU equations; file-generation main omitted."""
import torch

def make_fixtures():
    B, T, H, D, BT = (4, 2048, 8, 64, 64)
    rng = torch.Generator(device='cpu').manual_seed(20261030)
    k = torch.randn(B, T, H, D, generator=rng) * 0.25
    v = torch.randn(B, T, H, D, generator=rng) * 0.25
    beta = torch.rand(B, T, H, generator=rng)
    g_cumsum = (torch.rand(B, T, H, generator=rng) - 0.5) * 0.25
    A = torch.randn(B, T // BT, H, BT, BT, generator=rng) * 0.01
    A = (A.tril(-1) + torch.eye(BT)).permute(0, 1, 3, 2, 4).reshape(B, T, H, BT).contiguous()

    def blocks(x):
        return x.reshape(B, T // BT, BT, H, -1).permute(0, 1, 3, 2, 4)
    w = (blocks(A).double() @ blocks(k * beta.unsqueeze(-1) * g_cumsum.exp2().unsqueeze(-1)).double()).float().permute(0, 1, 3, 2, 4).reshape(B, T, H, D).contiguous()
    u = (blocks(A).double() @ blocks(v * beta.unsqueeze(-1)).double()).float().permute(0, 1, 3, 2, 4).reshape(B, T, H, D).contiguous()
    return {name + '.pt': value for name, value in dict(k=k, v=v, beta=beta, g_cumsum=g_cumsum, A=A, output_0=w, output_1=u).items()}
