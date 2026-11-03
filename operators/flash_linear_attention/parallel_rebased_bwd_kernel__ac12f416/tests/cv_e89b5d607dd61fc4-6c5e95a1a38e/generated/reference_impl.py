"""Existing independent CPU equations; file-generation main omitted."""
import math
import torch

def reference(cid, d):
    errors = []
    q, k, v = (d[n].double() for n in ['q', 'k', 'v'])
    assert d['NK'] == d['NV'] == 1
    score = torch.tril(q * d['scale'] @ k.transpose(-1, -2))
    do, dz = (d['do'].double(), d['dz'].double())
    grad = torch.tril(2 * score * (do @ v.transpose(-1, -2) + dz[..., None]) * d['scale'])
    dq = grad @ k
    dk = grad.transpose(-1, -2) @ q
    dv = score.square().transpose(-1, -2) @ do
    row = q.shape[-2] - 1
    expected = math.fsum((2 * math.fsum((float(q[0, 0, row, i]) * float(k[0, 0, j, i]) for i in range(d['K']))) * d['scale'] ** 2 * (math.fsum((float(do[0, 0, row, i]) * float(v[0, 0, j, i]) for i in range(d['V']))) + float(dz[0, 0, row])) * float(k[0, 0, j, 0]) for j in range(row + 1)))
    errors.append(abs(float(dq[0, 0, row, 0]) - expected))
    out = {'dq': dq.unsqueeze(0), 'dk': dk.unsqueeze(0), 'dv': dv.unsqueeze(0)}
    method = 'Independent CPU64 mathematical derivatives of causal squared scaled-qk attention; original FP16 input/output, no FP16 rounding of golden'
    assert all((x.dtype == torch.float64 and torch.isfinite(x).all() for x in out.values()))
    assert max(errors) < 1e-08, errors
    return (out, {'method': method, 'scalar_checked_values': len(errors), 'scalar_max_abs_difference': max(errors)})
