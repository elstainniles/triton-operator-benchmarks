"""Existing independent CPU equations; file-generation main omitted."""
import torch

def make_fixtures():
    torch.manual_seed(20261030)
    B, T, H, HQ, K, V, BS, S = (2, 1024, 8, 128, 64, 64, 64, 16)
    q = torch.randn(B, T, HQ, K, dtype=torch.float16)
    k = torch.randn(B, T, H, K, dtype=torch.float16)
    v = torch.randn(B, T, H, V, dtype=torch.float16)
    indices = torch.empty(B, T, H, S, dtype=torch.int32)
    for t in range(T):
        indices[:, t] = torch.randint(0, t // BS + 1, (B, H, S), dtype=torch.int32)
    o = torch.empty(B, T, HQ, V, dtype=torch.float16)
    lse = torch.empty(B, T, HQ, dtype=torch.float32)
    scaled = (q.double() * K ** (-0.5)).to(torch.float16)
    for b in range(B):
        for h in range(H):
            keys = k[b, :, h].double()
            vals = v[b, :, h].double()
            for start in range(0, T, 64):
                end = min(start + 64, T)
                n = end - start
                query = scaled[b, start:end, h * 16:(h + 1) * 16].permute(1, 0, 2).double()
                scores = query @ keys.T
                counts = torch.zeros(n, T // BS, dtype=torch.float64)
                counts.scatter_add_(1, indices[b, start:end, h].long(), torch.ones(n, S, dtype=torch.float64))
                counts = counts.repeat_interleave(BS, dim=1)
                counts.masked_fill_(torch.arange(T)[None, :] > torch.arange(start, end)[:, None], 0.0)
                scores += counts.log().unsqueeze(0)
                soft = torch.softmax(scores, dim=-1)
                out = (soft @ vals).permute(1, 0, 2).to(torch.float16)
                o[b, start:end, h * 16:(h + 1) * 16] = out
                lse[b, start:end, h * 16:(h + 1) * 16] = torch.logsumexp(scores, dim=-1).T.float()
    return {name + '.pt': value for name, value in dict(q=q, k=k, v=v, block_indices=indices, o=o, lse=lse).items()}
