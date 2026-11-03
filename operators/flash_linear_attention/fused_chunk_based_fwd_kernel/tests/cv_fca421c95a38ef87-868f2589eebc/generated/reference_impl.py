"""Existing independent CPU equations; file-generation main omitted."""
import math
import torch

def reference(cid, d):
    errors = []
    q, k, v = (d[n].double() for n in ['q', 'k', 'v'])
    b, h, t, width = q.shape
    assert width == 16, 'Original workload has one key tile; no full-key square substitution for NK>1'
    out = torch.empty_like(v)
    z = torch.empty(q.shape[:-1], dtype=torch.float64)
    keys = k.transpose(-1, -2)
    all_columns = torch.arange(t)
    for start in range(0, t, 128):
        end = min(start + 128, t)
        scores = q[:, :, start:end] * d['scale'] @ keys
        weights = 1 + scores + 0.5 * scores.square()
        weights.masked_fill_(all_columns[None, :] > torch.arange(start, end)[:, None], 0)
        out[:, :, start:end] = weights @ v
        z[:, :, start:end] = weights.sum(-1)
    for row, col in [(0, 0), (1, 1), (t - 1, v.shape[-1] - 1)]:
        terms = []
        mass = []
        for j in range(row + 1):
            dot = math.fsum((float(q[0, 0, row, ki]) * float(k[0, 0, j, ki]) for ki in range(width))) * d['scale']
            weight = 1 + dot + 0.5 * dot * dot
            mass.append(weight)
            terms.append(weight * float(v[0, 0, j, col]))
        errors.extend([abs(float(out[0, 0, row, col]) - math.fsum(terms)), abs(float(z[0, 0, row]) - math.fsum(mass))])
    outputs = {'o': out, 'z': z}
    method = 'Independent CPU64 causal Taylor attention: weights=1+(q*scale)@kT+0.5*((q*scale)@kT)^2; output=weights@v and z=sum(weights)'
    assert all((x.dtype == torch.float64 and torch.isfinite(x).all() for x in outputs.values()))
    assert max(errors) < 1e-09, errors
    return (outputs, {'method': method, 'scalar_checked_values': len(errors), 'scalar_max_abs_difference': max(errors)})
