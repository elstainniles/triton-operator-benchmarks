"""Existing independent CPU equations; file-generation main omitted."""
import torch

def build(kind):
    gen = torch.Generator().manual_seed(20261031)

    def rand(*s):
        return torch.randn(s, generator=gen) * 0.1
    if 'ttt_h'.startswith('oja'):
        B, T, H, D = (4, 4096, 4, 128)
        BT, BC, BV = (64, 16, 64)
        NT = T // BT
        shape = (B, T, H, D)
        d = dict(v=rand(*shape), g=rand(*shape), o=rand(*shape), A=rand(B, T, H, BT), do=rand(*shape), dv=rand(*shape), dv2=torch.zeros(shape), dg=torch.zeros(shape), cu_seqlens=None, chunk_indices=None, T=T, H=H, V=D, BT=BT, BC=BC, BV=BV, NC=BT // BC)
        grid = (D // BV, NT * (BT // BC), B * H)
        return dict(input_data=d, grid=grid)
    B, T, H, D, BT = (4, 2048, 8, 64, 16)
    NT = T // BT
    shape = (B, T, H, D)
    state = (B, H, D, D)
    d = dict(k=rand(*shape), v=rand(*shape), v2=torch.zeros(shape), x=torch.zeros(shape), y=torch.zeros(shape), r=torch.zeros(B, T, H, 1), w=1 + rand(H, D) * 0.5, b=rand(H, D), eta=rand(B, T, H, 1) * 0.05, h0=rand(*state), hb0=rand(B, H, 1, D), h=torch.zeros(B, NT, H, D, D), do=rand(*shape), dq=torch.zeros(shape), scale=D ** (-0.5), eps=1e-05, T=T, H=H, K=D, V=D, BT=BT, BK=D, BV=D)
    return dict(input_data=d, grid=(B * H,))
