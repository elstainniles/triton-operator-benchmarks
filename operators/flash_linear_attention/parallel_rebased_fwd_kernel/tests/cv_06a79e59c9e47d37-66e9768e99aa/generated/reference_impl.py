"""Existing independent CPU equations; file-generation main omitted."""
import math
import torch

def reference(cid, d):
    errors = []
    q, k, v = (d[n].double() for n in ['q', 'k', 'v'])
    assert d['NK'] == d['NV'] == 1
    score = torch.tril(q * d['scale'] @ k.transpose(-1, -2))
    weight = score.square()
    value = weight @ v
    out = {'o': value.unsqueeze(0), 'z': weight.sum(-1).unsqueeze(0)}
    row = q.shape[-2] - 1
    expected = math.fsum(((math.fsum((float(q[0, 0, row, i]) * float(k[0, 0, j, i]) for i in range(d['K']))) * d['scale']) ** 2 * float(v[0, 0, j, 0]) for j in range(row + 1)))
    errors.append(abs(float(value[0, 0, row, 0]) - expected))
    method = 'Independent CPU64 causal squared qk attention and normalization, preserving original one-key-tile partial outputs'
    assert all((x.dtype == torch.float64 and torch.isfinite(x).all() for x in out.values()))
    assert max(errors) < 1e-08, errors
    return (out, {'method': method, 'scalar_checked_values': len(errors), 'scalar_max_abs_difference': max(errors)})
