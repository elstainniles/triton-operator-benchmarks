"""Existing independent CPU equations; file-generation main omitted."""
import torch.nn.functional as F
import torch

def build(kind):
    seed = 42
    g = torch.Generator().manual_seed(seed)

    def r(shape, dtype=torch.float32):
        return (torch.randn(shape, generator=g) * 0.1).to(dtype)
    if 'wy64'.startswith('path'):
        B, T, H, HQ, D, S, BT, BS = (2, 2000, 1, 4, 64, 512, 128, 64)
        low = torch.float16
        sk = (B, T, H, D)
        sq = (B, T, HQ, D)
        d = dict(v=r(sk, low), g_cumsum=None, do=r(sq, low), dv=r(sq), dg_cumsum=None, scale=D ** (-0.5), L=torch.rand(B, T, HQ, generator=g) + 1, delta=r((B, T, HQ)), S=S)
        d.update(q_new=r(sq, low), k_new=r(sk, low), w=F.normalize(r(sk), dim=-1), h=r(sk), dw1=r(sq), dw2=r(sq), dq=r(sq), dk=r(sq), BT=BS)
        return (d, 'plain', seed, None)
    B, T, H, BT = (2, 512, 4, 64)
    blocks = r((B, T // BT, H, BT, BT)) * 0.1
    blocks = blocks.tril(-1)
    a = blocks.permute(0, 1, 3, 2, 4).contiguous().view(B, T, H, BT)
    return (dict(A_ab=a, cu_seqlens=None, chunk_size=BT), 'plain', seed, None)
