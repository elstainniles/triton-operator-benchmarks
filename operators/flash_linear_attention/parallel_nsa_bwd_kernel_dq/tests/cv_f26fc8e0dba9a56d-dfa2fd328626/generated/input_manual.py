"""Existing independent CPU equations; file-generation main omitted."""
import torch
SEED = 42

def raw_inputs(D):
    g = torch.Generator().manual_seed(SEED)
    B, T, H, HQ, S, BS = (3, 1024, 2, 32, 16, 32)
    q = torch.randn(B, T, HQ, D, dtype=torch.float16, generator=g)
    k = torch.randn(B, T, H, D, dtype=torch.float16, generator=g)
    v = torch.randn(B, T, H, D, dtype=torch.float16, generator=g)
    do = torch.randn(B, T, HQ, D, dtype=torch.float16, generator=g)
    indices = torch.full((B, T, H, S), T, dtype=torch.int64)
    for b in range(B):
        for t in range(T):
            for h in range(H):
                selected = torch.randperm(max(1, (t + BS - 1) // BS), generator=g)[:S]
                indices[b, t, h, :len(selected)] = selected
    indices = indices.sort(-1).values
    g_cmp = torch.zeros(B, T, HQ, dtype=torch.float32).uniform_(0.95, 1, generator=g).log()
    return (q, k, v, do, indices, g_cmp)

def attention(q, k, v, do, mask):
    """Dense CPU64 matrix equations with original input/explicit scaling casts."""
    B, T, HQ, D = q.shape
    H = k.shape[2]
    G = HQ // H
    TC = k.shape[1]
    scale = 0.1
    o = torch.zeros(B, T, HQ, D, dtype=q.dtype)
    lse = torch.zeros(B, T, HQ, dtype=torch.float32)
    dq = torch.zeros(B, T, HQ, D, dtype=torch.float64)
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

def build(kind):
    D = 60
    q, k, v, do, indices, g_cmp = raw_inputs(D)
    B, T, H, HQ, S, BS, G = (3, 1024, 2, 32, 16, 32, 16)
    blocks = torch.zeros(B, H, T, T // BS + 1, dtype=torch.bool)
    safe = indices.permute(0, 2, 1, 3).clamp_max(T // BS)
    blocks.scatter_(-1, safe, True)
    mask = blocks[..., :T // BS].repeat_interleave(BS, -1) & torch.ones(T, T, dtype=torch.bool).tril()[None, None]
    o, lse, delta, dq, checks = attention(q, k, v, do, mask)
    payload = dict(q=q, k=k, v=v, lse=lse, delta=delta, do=do, block_indices=indices, block_counts=S, scale=0.1, cu_seqlens=None, token_indices=None, B_param=B, T_param=T, H_param=H, HQ_param=HQ, G_param=G, K_param=D, V_param=D, S_param=S, BS_param=BS, BK_param=64, BV_param=64, NV_param=1)
    return (payload, dict(dq=dq), checks, 'inputs.pt', 'plain')
