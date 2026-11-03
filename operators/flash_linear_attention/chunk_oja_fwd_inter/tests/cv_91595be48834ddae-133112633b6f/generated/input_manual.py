"""Existing independent CPU equations; file-generation main omitted."""
import torch

def build(kind):
    gen = torch.Generator().manual_seed(20261031)

    def rand(*s):
        return torch.randn(s, generator=gen) * 0.1
    if 'oja_inter'.startswith('oja'):
        B, T, H, D = (2, 1024, 4, 128)
        BT, BC, BV = (64, 16, 64)
        NT = T // BT
        shape = (B, T, H, D)
        d = dict(q=rand(*shape), k=rand(*shape), h=rand(B, NT, H, D, D), gv=rand(*shape), o=torch.zeros(shape), A=torch.zeros(B, T, H, BT), cu_seqlens=None, chunk_indices=None, scale=D ** (-0.5), T=T, HQ=H, H=H, K=D, V=D, BT=BT, NG=1)
        grid = (D // BV, NT, B * H)
        return dict(input_data=d, grid=grid)
    B, T, H, D, BT = (4, 2048, 8, 64, 16)
    NT = T // BT
    shape = (B, T, H, D)
    state = (B, H, D, D)
    d = dict(q=rand(*shape), k=rand(*shape), v=rand(*shape), v2=rand(*shape), x=rand(*shape), y=rand(*shape), r=rand(B, T, H, 1).abs() + 1, w=rand(H, D), b=rand(H, D), eta=rand(B, T, H, 1) * 0.05, h=rand(B, NT, H, D, D), dht=rand(*state), dhbt=rand(B, H, 1, D), dh0=torch.zeros(state), dhb0=torch.zeros(B, H, 1, D), do=rand(*shape), dk=torch.zeros(shape), dv=torch.zeros(shape), de=torch.zeros(B, T, H, 1), dw=torch.zeros(B, H, D), db=torch.zeros(B, H, D), scale=D ** (-0.5), T=T, H=H, K=D, V=D, BT=BT, BK=D, BV=D)
    return dict(input_data=d, grid=(B * H,))
