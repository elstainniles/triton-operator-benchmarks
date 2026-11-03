"""Existing independent CPU equations; file-generation main omitted."""
import math
import torch

def reference(cid, d):
    checks = []
    notes = {}
    q, k, v, do, o, lse = (d[n].double() for n in ('q', 'k', 'v', 'do', 'o', 'lse'))
    bsz, length, hq, width = q.shape
    _, keys, hk, _ = k.shape
    groups = hq // hk
    assert v.shape[-1] <= 128
    dk, dv = (torch.zeros_like(k), torch.zeros_like(v))
    delta = (do * o).sum(-1)
    for b in range(bsz):
        for h in range(hk):
            qs = q[b, :, h * groups:(h + 1) * groups].reshape(length * groups, width) * d['scale']
            dos = do[b, :, h * groups:(h + 1) * groups].reshape(length * groups, v.shape[-1])
            ls = lse[b, :, h * groups:(h + 1) * groups].reshape(-1)
            de = delta[b, :, h * groups:(h + 1) * groups].reshape(-1)
            score = qs @ k[b, :, h].T
            p = torch.exp(score - ls[:, None])
            time = torch.arange(length).repeat_interleave(groups)
            mask = time[:, None] >= ((torch.arange(keys) + 1) * d['block_size'] - 1)[None, :]
            p = p.masked_fill(~mask, 0)
            ds = p * (dos @ v[b, :, h].T - de[:, None])
            dk[b, :, h] = ds.T @ qs
            dv[b, :, h] = p.T @ dos
            if b == 0 and h == 0:
                for key in (0, keys // 2, keys - 1):
                    expected_dv = math.fsum((float(p[j, key]) * float(dos[j, 0]) for j in range(length * groups)))
                    expected_dk = math.fsum((float(ds[j, key]) * float(qs[j, 0]) for j in range(length * groups)))
                    checks.extend([abs(float(dv[b, key, h, 0]) - expected_dv), abs(float(dk[b, key, h, 0]) - expected_dk)])
                j, key = (length * groups - 1, keys // 2)
                pv = math.exp(math.fsum((float(qs[j, z]) * float(k[b, key, h, z]) for z in range(width))) - float(ls[j])) if mask[j, key] else 0.0
                checks.append(abs(float(p[j, key]) - pv))
    out = {'dk': dk, 'dv': dv}
    method = 'Independent CPU64 grouped-query attention derivatives with complete-block causal compression mask'
    assert all((x.dtype == torch.float64 and torch.isfinite(x).all() for x in out.values()))
    assert checks and max(checks) < 1e-08, ('cv_1c7a483aee96b4e2', checks)
    return (out, {'method': method, 'scalar_checked_values': len(checks), 'scalar_max_abs_difference': max(checks), **notes})
