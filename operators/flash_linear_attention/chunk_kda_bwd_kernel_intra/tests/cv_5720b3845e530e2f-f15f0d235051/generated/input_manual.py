"""Existing independent CPU equations; file-generation main omitted."""
import torch

def build(kind):
    gen = torch.Generator().manual_seed(20261031)

    def r(shape, dtype=torch.float32):
        return (torch.randn(shape, generator=gen) * 0.1).to(dtype)
    B, T, H, HV, K, BT = (2, 500, 3, 3, 60, 64)
    shape = (B, T, HV, K)
    return dict(q=r((B, T, H, K), torch.float16), k=r((B, T, H, K), torch.float16), g=r(shape), beta=r((B, T, HV), torch.float16).sigmoid(), dAqk=r((B, T, HV, BT)), dAkk=r((B, T, HV, BT)), dq=r(shape), dk=r(shape), db=r((B, T, HV)), dg=r(shape), chunk_size=BT, safe_gate=False)
