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
    value = expand(tile(d['v'], bt))
    weight = a * beta[..., None, :]
    wp = expand(w)
    kp = expand(k)
    weightp = expand(weight)
    qw = torch.tril(q @ wp.transpose(-1, -2))
    qwt = qw @ weightp
    wk = torch.tril(w @ k.transpose(-1, -2), diagonal=-1)
    raw = torch.tril(q @ kp.transpose(-1, -2) - qwt @ expand(wk))
    query = q - qwt @ wp
    w2 = weight @ w
    key = k - (weight @ wk).transpose(-1, -2) @ w
    gate = scalar_tile(d['g_cumsum'], bt)
    logits = (raw + gate[..., None] - gate[..., None, :]) * (d['scale'] * 1.44269504)
    logits = logits.masked_fill(torch.ones(bt, bt, dtype=torch.bool).triu(1), float('-inf'))
    maximum = logits.amax(-1)
    prob = torch.exp2(logits - maximum[..., None])
    mass = prob.sum(-1)
    output = prob @ value
    for row, col in [(0, 0), (1, 1), (bt - 1, width - 1)]:
        expected = float(q[0, 0, 0, row, col]) - math.fsum((float(qwt[0, 0, 0, row, j]) * float(wp[0, 0, 0, j, col]) for j in range(bt)))
        errors.append(abs(float(query[0, 0, 0, row, col]) - expected))
    out = {'q_new': untile(query), 'k_new': untile(key), 'w2': untile(w2), 'o': untile(output), 'L': untile(mass[..., None]).squeeze(-1), 'M': untile(maximum[..., None]).squeeze(-1)}
    method = 'Independent CPU64 grouped-head preprocessing matrix equations and causal exp2 softmax numerator, normalization mass, and max'
    assert all((x.dtype == torch.float64 and torch.isfinite(x).all() for x in out.values()))
    assert max(errors) < 1e-09, errors
    return (out, {'method': method, 'scalar_checked_values': len(errors), 'scalar_max_abs_difference': max(errors)})
