"""Existing independent CPU equations; file-generation main omitted."""
import math
import torch

def reference(cid, d):
    errors = []
    q, k, v, u = (d[n].double().transpose(0, 1) for n in ['q', 'k', 'v', 'u'])
    score = q @ k.transpose(-1, -2) * d['qk_scale']
    prefix = d['T'] - d['C']
    mask = torch.arange(d['T'])[None, :] >= prefix + torch.arange(d['C'])[:, None]
    score = score.masked_fill(mask, -1000000.0)
    lse = torch.logsumexp(score * math.log(2), dim=-1) / math.log(2)
    probability = torch.exp2(score - lse[..., None])
    updated = u.clone()
    updated[:, prefix:] = v - d['beta'].double().transpose(0, 1)[..., None] * (probability[..., :prefix] @ u[:, :prefix])
    w = probability[..., prefix:].masked_fill(mask[:, prefix:], 0)
    for row, col in [(0, 0), (1, 1), (d['C'] - 1, d['D'] - 1)]:
        expected = float(v[0, row, col]) - float(d['beta'][row, 0]) * math.fsum((float(probability[0, row, j]) * float(u[0, j, col]) for j in range(prefix)))
        errors.append(abs(float(updated[0, prefix + row, col]) - expected))
    out = {'u': updated.transpose(0, 1), 'w': w.transpose(0, 1), 'lse': lse.transpose(0, 1)}
    method = 'Independent CPU64 strict-causal exp2 softmax, prefix-value residual update, suffix probability matrix and base2 logsumexp'
    assert all((v.dtype == torch.float64 and torch.isfinite(v).all() for v in out.values()))
    assert max(errors) < 1e-08, errors
    return (out, {'method': method, 'scalar_checked_values': len(errors), 'scalar_max_abs_difference': max(errors)})
