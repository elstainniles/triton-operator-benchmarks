"""Existing independent CPU equations; file-generation main omitted."""
from reference_impl_tile import tile
from reference_impl_scalar_tile import scalar_tile
import math
from reference_impl_untile import untile
import torch

def reference(cid, d):
    errors = []
    bt = d.get('BT', d['A'].shape[-1])
    batch, t, heads, width = d['k'].shape
    q = tile(d['q'], bt)
    hq = q.shape[2]
    groups = hq // heads
    k, w = (tile(d[n], bt) for n in ['k', 'w'])
    beta = scalar_tile(d['beta'], bt)
    a = tile(d['A'].reshape(-1)[:batch * t * heads * bt].reshape(batch, t, heads, bt), bt)
    expand = lambda x: x.repeat_interleave(groups, dim=2)
    k, w, beta, a = map(expand, [k, w, beta, a])
    w2 = expand(tile(d['w2'], bt))
    dq, dk, da, dw1, dw2 = (tile(d[n], bt) for n in ['dq', 'dk', 'dA_local', 'dw1', 'dw2'])
    wb = w * beta[..., None]
    qw = torch.tril(q @ w.transpose(-1, -2))
    wbk = torch.tril(wb @ k.transpose(-1, -2), diagonal=-1)
    twbk = a @ wbk
    dqw = torch.tril(-da @ twbk.transpose(-1, -2) - dq @ w2.transpose(-1, -2))
    dtwb = -qw.transpose(-1, -2) @ dq + dw2
    dt = dtwb @ wb.transpose(-1, -2)
    dwb = a.transpose(-1, -2) @ dtwb
    dqnew = dq + da @ k + dqw @ w
    dw = dw1 + dqw.transpose(-1, -2) @ q - twbk @ dk
    dtwbk = -qw.transpose(-1, -2) @ da - w @ dk.transpose(-1, -2)
    dt = dt + dtwbk @ wbk.transpose(-1, -2)
    dwbk = torch.tril(a.transpose(-1, -2) @ dtwbk, diagonal=-1)
    dwb = dwb + dwbk @ k
    dknew = dk + dwbk.transpose(-1, -2) @ wb + da.transpose(-1, -2) @ q
    inversegrad = torch.tril(-(a.transpose(-1, -2) @ torch.tril(dt, diagonal=-1) @ a.transpose(-1, -2)), diagonal=-1)
    dwb = dwb + inversegrad @ w
    dw = dw + inversegrad.transpose(-1, -2) @ wb + dwb * beta[..., None]
    dbeta = (dwb * w).sum(-1)
    for row, col in [(0, 0), (1, 1), (bt - 1, width - 1)]:
        expected = float(dq[0, 0, 0, row, col]) + math.fsum((float(da[0, 0, 0, row, j]) * float(k[0, 0, 0, j, col]) + float(dqw[0, 0, 0, row, j]) * float(w[0, 0, 0, j, col]) for j in range(bt)))
        errors.append(abs(float(dqnew[0, 0, 0, row, col]) - expected))
    out = {'dq_new': untile(dqnew), 'dk_new': untile(dknew), 'dbeta': untile(dbeta[..., None]).squeeze(-1), 'dw': untile(dw)}
    method = 'Independent CPU64 analytic matrix-chain derivatives, strict-lower inverse derivative and original grouped-head/contiguous A pointer semantics'
    assert all((x.dtype == torch.float64 and torch.isfinite(x).all() for x in out.values()))
    assert max(errors) < 1e-09, errors
    return (out, {'method': method, 'scalar_checked_values': len(errors), 'scalar_max_abs_difference': max(errors)})
