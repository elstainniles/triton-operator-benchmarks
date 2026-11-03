"""Existing independent CPU equations; file-generation main omitted."""
from reference_impl_tile import tile
from reference_impl_scalar_tile import scalar_tile
import math
from reference_impl_untile import untile
import torch

def reference(cid, d):
    bt = d.get('BT', d.get('chunk_size'))
    errors = []
    q, k, v = (tile(d[n], bt) for n in ['q', 'k', 'v'])
    batch, count, hv, _, vd = v.shape
    if q.shape[2] != hv:
        q = q.repeat_interleave(hv // q.shape[2], dim=2)
        k = k.repeat_interleave(hv // k.shape[2], dim=2)
    h = d['h'].double().reshape(batch, count, hv, q.shape[-1], vd)
    cross = q @ h
    attention = q @ k.transpose(-1, -2)
    gates = torch.zeros(q.shape[:-1], dtype=torch.float64)
    if d['g'] is not None:
        gates = gates + scalar_tile(d['g'], bt)
    if d['g_gamma'] is not None:
        gates = gates + d['g_gamma'].double()[None, None, :, None] * torch.arange(1, bt + 1, dtype=torch.float64)
    cross = cross * torch.exp2(gates[..., None])
    attention = torch.tril(attention * torch.exp2(gates[..., None] - gates[..., None, :]))
    out_t = (cross + attention @ v) * d['scale']
    for row, col in [(0, 0), (1, 1), (bt - 1, vd - 1)]:
        c = math.fsum((float(q[0, 0, 0, row, z]) * float(h[0, 0, 0, z, col]) for z in range(q.shape[-1]))) * 2.0 ** float(gates[0, 0, 0, row])
        l = math.fsum((math.fsum((float(q[0, 0, 0, row, z]) * float(k[0, 0, 0, j, z]) for z in range(q.shape[-1]))) * 2.0 ** float(gates[0, 0, 0, row] - gates[0, 0, 0, j]) * float(v[0, 0, 0, j, col]) for j in range(row + 1)))
        errors.append(abs(float(out_t[0, 0, 0, row, col]) - (c + l) * d['scale']))
    out = {'o': untile(out_t)}
    method = 'CPU64 gated q@h plus causal q@k^T@v, original head mapping and scale'
    assert all((v.dtype == torch.float64 and torch.isfinite(v).all() for v in out.values()))
    assert max(errors) < 1e-08, errors
    return (out, {'method': method, 'scalar_checked_values': len(errors), 'scalar_max_abs_difference': max(errors)})
