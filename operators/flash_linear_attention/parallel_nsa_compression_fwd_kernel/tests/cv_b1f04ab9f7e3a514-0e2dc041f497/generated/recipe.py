"""Existing independent CPU equations; file-generation main omitted."""
import torch

def make_fixtures():
    torch.manual_seed(20261030)
    B, T, H, HQ, K, V, BS = (2, 1024, 8, 32, 128, 128, 64)
    q = torch.randn(B, T, HQ, K, dtype=torch.float16)
    k = torch.randn(B, T // BS, H, K, dtype=torch.float16)
    v = torch.randn(B, T // BS, H, V, dtype=torch.float16)
    o = torch.zeros(B, T, HQ, V, dtype=torch.float16)
    lse = torch.zeros(B, T, HQ, dtype=torch.float32)
    scaled = (q.double() * K ** (-0.5)).to(torch.float16)
    for b in range(B):
        for hq in range(HQ):
            h = hq // (HQ // H)
            scores = scaled[b, :, hq].double() @ k[b, :, h].double().T
            nc = (torch.arange(T) + 1) // BS
            mask = torch.arange(T // BS)[None, :] >= nc[:, None]
            scores.masked_fill_(mask, float('-inf'))
            good = nc > 0
            s = scores[good]
            o[b, good, hq] = (torch.softmax(s, dim=-1) @ v[b, :, h].double()).to(torch.float16)
            lse[b, good, hq] = torch.logsumexp(s, dim=-1).float()
    return {name + '.pt': value for name, value in dict(q=q, k=k, v=v, o=o, lse=lse).items()}
