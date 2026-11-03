"""Existing independent CPU equations; file-generation main omitted."""
import torch

def attention(q, k, v, do, mask):
    """Dense CPU64 matrix equations with original input/explicit scaling casts."""
    B, T, HQ, D = q.shape
    H = k.shape[2]
    G = HQ // H
    TC = k.shape[1]
    scale = 0.1
    o = torch.zeros(B, T, HQ, D, dtype=q.dtype)
    lse = torch.zeros(B, T, HQ, dtype=torch.float32)
    dq = torch.empty(B, T, HQ, D, dtype=torch.float64)
    checks = []
    for b in range(B):
        for h in range(H):
            hs = slice(h * G, (h + 1) * G)
            qm = (q[b, :, hs].float() * scale).to(q.dtype).double().transpose(0, 1)
            km = k[b, :, h].double()
            vm = v[b, :, h].double()
            dom = do[b, :, hs].double().transpose(0, 1)
            valid = mask[b, h] if mask.ndim == 4 else mask
            scores = qm @ km.T
            masked = scores.masked_fill(~valid[None], float('-inf'))
            ll = torch.logsumexp(masked, -1)
            available = valid.any(-1)
            ll[:, ~available] = 0
            pp = torch.exp(scores - ll[..., None]).masked_fill(~valid[None], 0)
            oo = (pp @ vm).transpose(0, 1).to(q.dtype)
            o[b, :, hs] = oo
            lse[b, :, hs] = ll.T.float()
            delta = (oo.float() * do[b, :, hs].float()).sum(-1)
            prob = torch.exp(scores - lse[b, :, hs].T.double()[..., None]).masked_fill(~valid[None], 0)
            dp = dom @ vm.T
            ds = (prob * (dp - delta.T.double()[..., None])).to(k.dtype).double()
            dd = ds @ km * scale
            dq[b, :, hs] = dd.transpose(0, 1)
            for t in (0, 31, 1023):
                alternate = torch.einsum('j,jd->d', ds[0, t], km) * scale
                err = float((alternate - dd[0, t]).abs().max())
                assert err < 1e-10
                checks.append(dict(b=b, h=h, t=t, max_abs_error=err))
    return (o, lse, (o.float() * do.float()).sum(-1), dq, checks)
