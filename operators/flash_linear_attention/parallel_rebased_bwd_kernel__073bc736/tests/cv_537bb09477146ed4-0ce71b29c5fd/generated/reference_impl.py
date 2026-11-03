"""Existing independent CPU equations; file-generation main omitted."""
import math
import torch

def reference(cid, d):
    errors = []
    q, k, v, do, dz = (d[n].double() for n in ['q', 'k', 'v', 'do', 'dz'])
    assert d['BK'] == d['K'] == 64 and d['BV'] == d['V'] == 64
    score = torch.tril(q * d['scale'] @ k.transpose(-1, -2))
    grad = torch.tril(2 * score * (do @ v.transpose(-1, -2) + dz[..., None]) * d['scale'])
    dk = grad.transpose(-1, -2) @ q
    dv = score.square().transpose(-1, -2) @ do
    for row, col in [(0, 0), (1, 1), (q.shape[-2] - 1, q.shape[-1] - 1)]:
        expected = math.fsum(((math.fsum((float(q[0, 0, j, i]) * float(k[0, 0, row, i]) for i in range(d['K']))) * d['scale']) ** 2 * float(do[0, 0, j, col]) for j in range(row, q.shape[-2])))
        errors.append(abs(float(dv[0, 0, row, col]) - expected))
    out = {'dk': dk.unsqueeze(0), 'dv': dv.unsqueeze(0), 'dq': d['dq'].double().clone()}
    method = 'Independent CPU64 causal squared-attention derivatives for k/v; original dq argument is not written and preserved'
    assert all((x.dtype == torch.float64 and torch.isfinite(x).all() for x in out.values()))
    assert max(errors) < 1e-09, errors
    return (out, {'method': method, 'scalar_checked_values': len(errors), 'scalar_max_abs_difference': max(errors)})
