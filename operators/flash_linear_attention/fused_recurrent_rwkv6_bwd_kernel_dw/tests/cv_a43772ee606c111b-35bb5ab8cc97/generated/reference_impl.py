"""Existing independent CPU equations; file-generation main omitted."""
import math
import torch

def reference(cid, d):
    errors = []
    q, k, dq, dk = (d[n].double() for n in ['q', 'k', 'dq1', 'dk1'])
    delta = torch.zeros_like(q)
    delta[:, :-1] = q[:, 1:] * dq[:, 1:] * d['scale'] - k[:, :-1] * dk[:, :-1]
    value = delta.cumsum(1) if d['reverse'] else delta.flip(1).cumsum(1).flip(1)
    for t, h, ki in [(0, 0, 0), (1, 1, 1), (q.shape[1] - 1, q.shape[2] - 1, q.shape[-1] - 1)]:
        times = range(t + 1) if d['reverse'] else range(t, q.shape[1])
        expected = math.fsum((float(delta[0, i, h, ki]) for i in times))
        errors.append(abs(float(value[0, t, h, ki]) - expected))
    out = {'dw': value}
    method = 'CPU64 shifted q*dq*scale-k*dk prefix/reverse-prefix sum with original wrapper REVERSE=not reverse; original dtype-specific acceptance retained'
    assert all((v.dtype == torch.float64 and torch.isfinite(v).all() for v in out.values()))
    assert max(errors) < 1e-08, errors
    return (out, {'method': method, 'scalar_checked_values': len(errors), 'scalar_max_abs_difference': max(errors)})
