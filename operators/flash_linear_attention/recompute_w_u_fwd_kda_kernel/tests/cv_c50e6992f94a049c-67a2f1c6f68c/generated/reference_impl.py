"""Existing independent CPU equations; file-generation main omitted."""
from reference_impl_tile import tile
from reference_impl_scalar_tile import scalar_tile
import math
from reference_impl_untile import untile
import torch

def reference(cid, d):
    errors = []
    bt = d['A'].shape[-1]
    k, v, a, g = (tile(d[n], bt) for n in ['k', 'v', 'A', 'gk'])
    beta = scalar_tile(d['beta'], bt)
    k = k.repeat_interleave(v.shape[2] // k.shape[2], dim=2)
    w = a @ (k * beta[..., None] * torch.exp2(g))
    u = a @ (v * beta[..., None])
    kg = k * torch.exp2(g[..., -1:, :] - g)
    out = {'w': untile(w), 'u': untile(u), 'kg': untile(kg)}
    assert d['q'] is None
    for row, col in [(0, 0), (1, 1), (bt - 1, k.shape[-1] - 1)]:
        expected = math.fsum((float(a[0, 0, 0, row, j]) * float(k[0, 0, 0, j, col]) * float(beta[0, 0, 0, j]) * 2.0 ** float(g[0, 0, 0, j, col]) for j in range(bt)))
        errors.append(abs(float(w[0, 0, 0, row, col]) - expected))
    method = 'CPU64 grouped-head KDA w/u/decayed-key projections, preserving q=None and all original tiles'
    assert all((x.dtype == torch.float64 and torch.isfinite(x).all() for x in out.values()))
    assert max(errors) < 1e-08, errors
    return (out, {'method': method, 'scalar_checked_values': len(errors), 'scalar_max_abs_difference': max(errors)})
