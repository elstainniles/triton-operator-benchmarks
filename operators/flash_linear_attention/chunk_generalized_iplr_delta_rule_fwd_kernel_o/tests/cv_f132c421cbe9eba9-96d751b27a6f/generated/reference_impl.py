"""Existing independent CPU equations; file-generation main omitted."""
import math
import torch

def tile(t, bt):
    b, n, h, d = t.shape
    assert n % bt == 0
    return t.double().reshape(b, n // bt, bt, h, d).permute(0, 1, 3, 2, 4)

def untile(t):
    b, n, h, bt, d = t.shape
    return t.permute(0, 1, 3, 2, 4).reshape(b, n * bt, h, d)

def reference(cid, d):
    bt = d.get('BT', d.get('chunk_size'))
    scalar_checks = []
    q, k, b, v, u = (tile(d[n], bt) for n in ['q', 'k', 'b', 'v', 'v_new'])
    batch, count, heads, _, width = q.shape
    h = d['h'].double().reshape(batch, count, heads, width, v.shape[-1])
    cross = q @ h
    aqk = torch.tril(q @ k.transpose(-1, -2))
    aqb = torch.tril(q @ b.transpose(-1, -2))
    out = (cross + aqk @ v + aqb @ u) * d['scale']
    for row, col in [(0, 0), (1, 1), (bt - 1, v.shape[-1] - 1)]:
        cross_s = math.fsum((float(q[0, 0, 0, row, kd]) * float(h[0, 0, 0, kd, col]) for kd in range(width)))
        local = 0.0
        for j in range(row + 1):
            a1 = math.fsum((float(q[0, 0, 0, row, kd]) * float(k[0, 0, 0, j, kd]) for kd in range(width)))
            a2 = math.fsum((float(q[0, 0, 0, row, kd]) * float(b[0, 0, 0, j, kd]) for kd in range(width)))
            local += a1 * float(v[0, 0, 0, j, col]) + a2 * float(u[0, 0, 0, j, col])
        scalar_checks.append(abs(float(out[0, 0, 0, row, col]) - (cross_s + local) * d['scale']))
    outputs = {'o': untile(out)}
    method = 'q @ h + tril(q @ k^T) @ v + tril(q @ b^T) @ v_new, multiplied by scale'
    assert all((v.dtype == torch.float64 and torch.isfinite(v).all() for v in outputs.values()))
    assert max(scalar_checks) < 1e-09, scalar_checks
    return (outputs, {'method': method, 'scalar_checked_values': len(scalar_checks), 'scalar_max_abs_difference': max(scalar_checks)})
