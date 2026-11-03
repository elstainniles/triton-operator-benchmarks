"""Existing independent CPU equations; file-generation main omitted."""
import torch

def build(kind):
    gen = torch.Generator().manual_seed(20261031)

    def r(shape, dtype=torch.float32):
        return (torch.randn(shape, generator=gen) * 0.1).to(dtype)
    B, T, H, HV, K = (4, 1024, 16, 32, 128)
    return dict(q=r((B, T, H, K)), k=r((B, T, H, K)), gk=r((B, T, HV, K)), beta=torch.rand(B, T, HV, generator=gen))
