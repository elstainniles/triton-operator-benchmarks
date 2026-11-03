"""Existing independent CPU equations; file-generation main omitted."""
from reference_impl_tile import tile
import math
from reference_impl_untile import untile
import torch

def reference(cid, d):
    bt = d.get('BT', d.get('chunk_size'))
    errors = []
    k, v, gv, dk = (tile(d[n], bt) for n in ['k', 'v', 'gv', 'dk'])
    batch, count, heads, _, width = k.shape
    h, dh = (d[n].double().reshape(batch, count, heads, width, v.shape[-1]) for n in ['h', 'dh'])
    assert d['gv'] is not None
    dv = k @ dh * torch.exp(gv[..., -1:, :] - gv)
    dw = -(dk @ h)
    last = (h * dh).sum(-2) * torch.exp(gv[..., -1, :]) + (dv * v).sum(-2)
    dg = last[..., None, :].expand_as(v)
    if d['dgk'] is not None:
        dg = dg + tile(d['dgk'], bt)
    for row, col in [(0, 0), (1, 1), (bt - 1, v.shape[-1] - 1)]:
        a = math.fsum((float(k[0, 0, 0, row, i]) * float(dh[0, 0, 0, i, col]) for i in range(width))) * math.exp(float(gv[0, 0, 0, -1, col] - gv[0, 0, 0, row, col]))
        b = -math.fsum((float(dk[0, 0, 0, row, i]) * float(h[0, 0, 0, i, col]) for i in range(width)))
        c = math.fsum((float(h[0, 0, 0, i, col]) * float(dh[0, 0, 0, i, col]) for i in range(width))) * math.exp(float(gv[0, 0, 0, -1, col])) + math.fsum((float(dv[0, 0, 0, t, col]) * float(v[0, 0, 0, t, col]) for t in range(bt)))
        if d['dgk'] is not None:
            c += float(tile(d['dgk'], bt)[0, 0, 0, row, col])
        errors.extend([abs(float(dv[0, 0, 0, row, col]) - a), abs(float(dw[0, 0, 0, row, col]) - b), abs(float(dg[0, 0, 0, row, col]) - c)])
    out = {'dv': untile(dv), 'dw': untile(dw), 'dgv_last': untile(dg)}
    method = 'CPU64 k@dh with value gates, -dk@h, and complete last-gate state plus value reduction'
    assert all((v.dtype == torch.float64 and torch.isfinite(v).all() for v in out.values()))
    assert max(errors) < 1e-08, errors
    return (out, {'method': method, 'scalar_checked_values': len(errors), 'scalar_max_abs_difference': max(errors)})
