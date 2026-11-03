"""Existing independent CPU equations; file-generation main omitted."""
import torch

def build(kind):
    gen = torch.Generator().manual_seed(20261031)

    def r(shape, dtype=torch.float32):
        return (torch.randn(shape, generator=gen) * 0.1).to(dtype)
    B, T, H, HV, K, V, BT = (3, 1024, 4, 4, 100, 100, 64)
    shape = (B, T, HV, V)
    return dict(q=r((B, T, H, K), torch.float16), k=r((B, T, H, K), torch.float16), v=r(shape, torch.float16), v_new=r(shape, torch.float16), g=r((B, T, HV, K)), beta=r((B, T, HV), torch.float16).sigmoid(), A=r((B, T, HV, BT), torch.float16), h=r((B, T // BT, HV, K, V), torch.float16), do=r(shape, torch.float16), dh=r((B, T // BT, HV, K, V), torch.float16), dv=r(shape, torch.float16), state_v_first=False, scale=1, chunk_size=BT)
