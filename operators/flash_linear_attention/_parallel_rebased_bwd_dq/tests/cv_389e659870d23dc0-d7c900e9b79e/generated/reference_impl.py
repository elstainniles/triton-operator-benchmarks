"""Existing independent CPU equations; file-generation main omitted."""
import math
import torch

def reference(cid, d):
    q, k, v, do, dz = (d[n].double() for n in ['q', 'k', 'v', 'do', 'dz'])
    assert d['BK'] == d['K'] == 64 and d['BV'] == d['V'] == 64
    score = q * d['scale'] @ k.transpose(-1, -2)
    upstream = do @ v.transpose(-1, -2) + dz[..., None]
    derivative = torch.tril(2 * score * upstream) * d['scale']
    dq = derivative @ k
    errors = []
    for row, col in [(0, 0), (1, 1), (q.shape[-2] - 1, q.shape[-1] - 1)]:
        expected = math.fsum((2 * math.fsum((float(q[0, 0, row, i]) * float(k[0, 0, j, i]) for i in range(d['K']))) * d['scale'] ** 2 * (math.fsum((float(do[0, 0, row, i]) * float(v[0, 0, j, i]) for i in range(d['V']))) + float(dz[0, 0, row])) * float(k[0, 0, j, col]) for j in range(row + 1)))
        errors.append(abs(float(dq[0, 0, row, col]) - expected))
    assert torch.isfinite(dq).all() and max(errors) < 1e-09
    return ({'dq': dq.unsqueeze(0)}, {'method': 'CPU64 derivative of causal squared scaled qk attention and normalization, whole dq output covering all 256 original disjoint launch tiles', 'scalar_checked_values': len(errors), 'scalar_max_abs_difference': max(errors)})
