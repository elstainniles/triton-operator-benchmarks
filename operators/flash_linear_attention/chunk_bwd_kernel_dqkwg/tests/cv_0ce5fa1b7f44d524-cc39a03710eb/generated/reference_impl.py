"""Existing independent CPU equations; file-generation main omitted."""
from reference_impl_tile import tile
import math
from reference_impl_untile import untile
import torch

def reference(cid, d):
    checks = []
    notes = {}
    assert all((d[n] is None for n in ('w', 'g', 'g_gamma', 'dv')))
    bt = d['chunk_size']
    q, k, v, do = (tile(d[n], bt) for n in ('q', 'k', 'v', 'do'))
    bsz, nt, hv, _, width = q.shape
    assert d['H'] == d['HV']
    h = d['h'].double().reshape(bsz, nt, hv, width, v.shape[-1])
    dh = d['dh'].double().reshape_as(h)
    ds = (do @ v.transpose(-1, -2)).tril()
    dq = (do @ h.transpose(-1, -2) + ds @ k) * d['scale']
    dk = v @ dh.transpose(-1, -2) + ds.transpose(-1, -2) @ q * d['scale']
    for row, col in ((0, 0), (1, 1), (bt - 1, width - 1)):
        first = math.fsum((float(do[0, 0, 0, row, j]) * float(h[0, 0, 0, col, j]) for j in range(v.shape[-1])))
        second = math.fsum((math.fsum((float(do[0, 0, 0, row, j]) * float(v[0, 0, 0, t, j]) for j in range(v.shape[-1]))) * float(k[0, 0, 0, t, col]) for t in range(row + 1)))
        checks.append(abs(float(dq[0, 0, 0, row, col]) - (first + second) * d['scale']))
        first = math.fsum((float(v[0, 0, 0, row, j]) * float(dh[0, 0, 0, col, j]) for j in range(v.shape[-1])))
        second = math.fsum((math.fsum((float(do[0, 0, 0, t, j]) * float(v[0, 0, 0, row, j]) for j in range(v.shape[-1]))) * float(q[0, 0, 0, t, col]) for t in range(row, bt)))
        checks.append(abs(float(dk[0, 0, 0, row, col]) - (first + second * d['scale'])))
    out = {'dq': untile(dq), 'dk': untile(dk)}
    method = 'Independent CPU64 chunk causal matrix derivatives dq=(do h^T+tril(do v^T) k)*scale; dk=v dh^T+tril(do v^T)^T q*scale'
    notes.update(null_outputs=['dw', 'dg'], transpose_state_layout=False)
    assert all((x.dtype == torch.float64 and torch.isfinite(x).all() for x in out.values()))
    assert checks and max(checks) < 1e-08, ('cv_0ce5fa1b7f44d524', checks)
    return (out, {'method': method, 'scalar_checked_values': len(checks), 'scalar_max_abs_difference': max(checks), **notes})
