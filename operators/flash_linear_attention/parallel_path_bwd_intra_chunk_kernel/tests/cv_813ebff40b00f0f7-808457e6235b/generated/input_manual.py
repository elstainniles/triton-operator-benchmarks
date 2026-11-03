"""Existing independent CPU equations; file-generation main omitted."""
import torch.nn.functional as F
import torch

def build(kind):
    seed = 42
    g = torch.Generator().manual_seed(seed)

    def r(shape, dtype=torch.float32):
        return (torch.randn(shape, generator=g) * 0.1).to(dtype)
    if 'path_intra'.startswith('path'):
        B, T, H, HQ, D, S, BT, BS = (2, 2000, 1, 4, 64, 512, 128, 64)
        low = torch.float16
        sk = (B, T, H, D)
        sq = (B, T, HQ, D)
        d = dict(v=r(sk, low), g_cumsum=None, do=r(sq, low), dv=r(sq), dg_cumsum=None, scale=D ** (-0.5), L=torch.rand(B, T, HQ, generator=g) + 1, delta=r((B, T, HQ)), S=S)
        d.update(q_new=r(sq, low), k_new=r(sk, low), w=F.normalize(r(sk), dim=-1), h=r(sk), dw1=r(sq), dw2=r(sq), dq=r(sq), dk=r(sq), BT=BS)
        return (d, 'plain', seed, None)
    B, H, T, K, V = (2, 4, 512, 16, 64)
    fp16 = torch.float16
    d = dict(q=r((B, H, T, K), fp16), k=r((B, H, T, K), fp16), v=r((B, H, T, V), fp16), do=r((B, H, T, V), fp16), dz=r((B, H, T), fp16), B=B, H=H, T=T, K=K, V=V, scale=0.1)
    BTL, BTS, BK, BV = (64, 32, 16, 64)
    NK = NV = 1
    d.update(dq=torch.zeros(NV, B, H, T, K), dk=torch.zeros(NV, B, H, T, K), dv=torch.zeros(NK, B, H, T, V), BTL=BTL, BTS=BTS, BK=BK, BV=BV)
    return (d, 'input_data', seed, (NK * NV, (T + BTL - 1) // BTL, B * H))
