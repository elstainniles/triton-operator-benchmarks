"""Existing independent CPU equations; file-generation main omitted."""
import torch.nn.functional as F
import torch

def make_fixtures():
    B, T, H, D, BT = (4, 2048, 8, 64, 16)
    N = T // BT
    rng = torch.Generator(device='cpu').manual_seed(20261030)
    q = torch.randn(B, T, H, D, generator=rng, dtype=torch.float16)
    k = F.normalize(torch.randn(B, T, H, D, generator=rng, dtype=torch.float32), dim=-1).half()
    v = torch.randn(B, T, H, D, generator=rng, dtype=torch.float16)
    eta = torch.randn(B, T, H, 1, generator=rng, dtype=torch.float16) * 0.005
    h = torch.randn(B, N, H, D, D, generator=rng, dtype=torch.float16) * 0.1
    hb = torch.randn(B, N, H, 1, D, generator=rng, dtype=torch.float16) * 0.1

    def split(x):
        return x.reshape(B, N, BT, H, D).permute(0, 1, 3, 2, 4)
    qs, ks, vs = map(split, (q, k, v))
    es = eta.reshape(B, N, BT, H).permute(0, 1, 3, 2)
    qk = (qs.double() @ ks.double().transpose(-1, -2)).float()
    lower = torch.tril(torch.ones(BT, BT, dtype=torch.bool))
    qk = qk.where(lower, 0).half()
    weights = (es.unsqueeze(-1) * qk).half()
    weights_bias = es.unsqueeze(-1).expand_as(qk).where(lower, 0).half()
    qh = (qs.double() @ h.double()).float()
    att = (weights.double() @ vs.double()).float()
    bias = (weights_bias.double() @ vs.double()).float()
    result = ((qh - att) * 0.125 + hb.float() - bias).half()
    result = result.permute(0, 1, 3, 2, 4).contiguous().reshape(B, T, H, D)
    return dict([('q.pt', q), ('k.pt', k), ('v_new.pt', v), ('eta.pt', eta), ('h.pt', h), ('hb.pt', hb), ('output.pt', result)])
