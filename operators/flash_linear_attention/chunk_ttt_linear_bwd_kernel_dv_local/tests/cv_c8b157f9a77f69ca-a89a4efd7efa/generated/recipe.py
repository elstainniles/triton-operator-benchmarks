"""Existing independent CPU equations; file-generation main omitted."""
import torch.nn.functional as F
import torch

def make_fixtures():
    B, T, H, D, BT = (4, 2048, 8, 64, 16)
    rng = torch.Generator(device='cpu').manual_seed(20261030)
    q = torch.randn(B, T, H, D, generator=rng, dtype=torch.float16)
    k = F.normalize(torch.randn(B, T, H, D, generator=rng, dtype=torch.float32), dim=-1).half()
    eta = torch.randn(B, T, H, 1, generator=rng, dtype=torch.float16) * 0.005
    do = torch.randn(B, T, H, D, generator=rng, dtype=torch.float16)

    def split(x):
        return x.reshape(B, T // BT, BT, H, x.shape[-1]).permute(0, 1, 3, 2, 4)
    qs, ks, ds = map(split, (q, k, do))
    es = split(eta).squeeze(-1)
    scores = (ks.double() @ qs.double().transpose(-1, -2)).float()
    upper = torch.triu(torch.ones(BT, BT, dtype=torch.bool))
    first = -(scores * 0.125 * es.float().unsqueeze(-2)).where(upper, 0).half()
    second = -es.unsqueeze(-2).expand_as(first).where(upper, 0).half()
    result = ((first.double() @ ds.double()).float() + (second.double() @ ds.double()).float()).half()
    result = result.permute(0, 1, 3, 2, 4).contiguous().reshape(B, T, H, D)
    return dict([('q.pt', q), ('k.pt', k), ('eta.pt', eta), ('do_tensor.pt', do), ('output.pt', result)])
