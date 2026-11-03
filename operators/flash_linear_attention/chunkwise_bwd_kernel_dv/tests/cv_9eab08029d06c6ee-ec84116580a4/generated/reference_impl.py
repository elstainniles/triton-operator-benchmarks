"""Existing independent CPU equations; file-generation main omitted."""
from reference_impl_tile import tile
from reference_impl_scalar_tile import scalar_tile
import math
from reference_impl_untile import untile
import torch

def reference(cid, d):
    bt = d['BT']
    errors = []
    k = tile(d['k'], bt).expand(-1, -1, d['H'], -1, -1)
    g = scalar_tile(d['g'], bt)
    dh = d['dh'].double()
    value = k @ dh * torch.exp(g[..., -1, None] - g)[..., None]
    for row, col in [(0, 0), (1, 1), (bt - 1, d['V'] - 1)]:
        expected = math.fsum((float(k[0, 0, 0, row, i]) * float(dh[0, 0, 0, i, col]) for i in range(d['K']))) * math.exp(float(g[0, 0, 0, -1] - g[0, 0, 0, row]))
        errors.append(abs(float(value[0, 0, 0, row, col]) - expected))
    out = {'dv': untile(value)}
    method = 'Independent CPU64 dv=exp(g_last-g)*(k@dh), shared key head expanded by original pointer contract'
    assert all((v.dtype == torch.float64 and torch.isfinite(v).all() for v in out.values()))
    assert max(errors) < 1e-09, errors
    return (out, {'method': method, 'scalar_checked_values': len(errors), 'scalar_max_abs_difference': max(errors)})
