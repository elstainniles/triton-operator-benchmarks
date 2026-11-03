"""Existing independent CPU equations; file-generation main omitted."""
import torch.nn.functional as F
import torch

def make_fixtures():
    B, T, H, D, BT = (4, 2048, 8, 64, 16)
    N = T // BT
    rng = torch.Generator(device='cpu').manual_seed(20261030)
    k = F.normalize(torch.randn(B, T, H, D, generator=rng), dim=-1).half()
    v = torch.randn(B, T, H, D, generator=rng, dtype=torch.float16) * 0.1
    w = torch.randn(H, D, generator=rng, dtype=torch.float16) * 0.1
    b = torch.randn(H, D, generator=rng, dtype=torch.float16) * 0.1
    eta = torch.randn(B, T, H, 1, generator=rng, dtype=torch.float16) * 0.005
    h0 = torch.randn(B, H, D, D, generator=rng) * 5.0
    hb0 = torch.randn(B, H, 1, D, generator=rng) * 5.0
    hs = torch.empty(B, N, H, D, D, dtype=torch.float16)
    hbs = torch.empty(B, N, H, 1, D, dtype=torch.float16)
    vnew = torch.empty_like(v)
    state = h0.clone()
    bias = hb0.clone()
    for c in range(N):
        hs[:, c] = state.half()
        hbs[:, c] = bias.half()
        kc = k[:, c * BT:(c + 1) * BT].permute(0, 2, 1, 3)
        vc = v[:, c * BT:(c + 1) * BT].permute(0, 2, 1, 3)
        activ = (kc.double() @ state.half().double()).float() + bias
        mean = activ.mean(-1, keepdim=True)
        centered = activ - mean
        rstd = torch.rsqrt((centered * centered).mean(-1, keepdim=True) + 1e-06)
        norm = centered * rstd
        u = (((norm.half() * w[None, :, None, :]).half() + b[None, :, None, :]).half() - vc).half()
        u = (u + kc).half()
        u = (u * w[None, :, None, :]).half()
        delta = rstd * (D * u.float() - u.float().sum(-1, keepdim=True) - norm.half().float() * (u * norm.half()).half().float().sum(-1, keepdim=True)) / D
        vnew[:, c * BT:(c + 1) * BT] = delta.half().permute(0, 2, 1, 3)
        last = eta[:, (c + 1) * BT - 1].permute(0, 1, 2).unsqueeze(-1)
        weighted = (last * kc).half()
        state = state - (weighted.transpose(-1, -2).double() @ delta.half().double()).float()
        bias = bias - (last * delta.half()).half().float().sum(-2, keepdim=True)
    return {name + '.pt': value.contiguous() for name, value in dict(k=k, v=v, w=w, b=b, eta=eta, h0=h0, hb0=hb0, output_0=hs, output_1=hbs, output_2=vnew, output_3=state, output_4=bias).items()}
