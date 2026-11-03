"""Existing independent CPU equations; file-generation main omitted."""
from reference_impl_tile import tile
from reference_impl_scalar_tile import scalar_tile
import math
from reference_impl_untile import untile
import torch

def reference(cid, d):
    errors = []
    bt = 64
    q, k, do = (tile(d[n], bt) for n in ['q', 'k', 'do'])
    g = scalar_tile(d['g'], bt)
    dh = d['dh'].double()
    a = torch.triu(k @ q.transpose(-1, -2) * torch.exp(g[..., None, :] - g[..., None]) * d['scale'])
    value = k @ dh * torch.exp(g[..., -1, None] - g)[..., None] + a @ do
    for row, col in [(0, 0), (1, 1), (bt - 1, do.shape[-1] - 1)]:
        expected = math.fsum((float(k[0, 0, 0, row, i]) * float(dh[0, 0, 0, i, col]) for i in range(k.shape[-1]))) * math.exp(float(g[0, 0, 0, -1] - g[0, 0, 0, row]))
        expected += math.fsum((float(a[0, 0, 0, row, j]) * float(do[0, 0, 0, j, col]) for j in range(row, bt)))
        errors.append(abs(float(value[0, 0, 0, row, col]) - expected))
    out = {'dv': untile(value)}
    method = 'CPU64 gated local-attention value gradient plus terminal-state contribution, original natural exp choice'
    assert all((v.dtype == torch.float64 and torch.isfinite(v).all() for v in out.values()))
    assert max(errors) < 1e-08, errors
    return (out, {'method': method, 'scalar_checked_values': len(errors), 'scalar_max_abs_difference': max(errors)})
