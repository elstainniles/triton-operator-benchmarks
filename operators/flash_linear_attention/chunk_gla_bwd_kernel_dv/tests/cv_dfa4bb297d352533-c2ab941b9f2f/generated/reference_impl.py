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
    k, g, a, do = (tile(d[n], bt) for n in ['k', 'g', 'A', 'do'])
    batch, count, heads, _, width = k.shape
    dh = d['dh'].double().reshape(batch, count, heads, width, do.shape[-1])
    kg = k * torch.exp2(g[..., -1:, :] - g)
    out = torch.tril(a).transpose(-1, -2) @ do + kg @ dh
    for row, col in [(0, 0), (1, 1), (bt - 1, do.shape[-1] - 1)]:
        local = math.fsum((float(a[0, 0, 0, j, row]) * float(do[0, 0, 0, j, col]) for j in range(row, bt)))
        cross = math.fsum((float(k[0, 0, 0, row, kd]) * 2.0 ** float(g[0, 0, 0, -1, kd] - g[0, 0, 0, row, kd]) * float(dh[0, 0, 0, kd, col]) for kd in range(width)))
        scalar_checks.append(abs(float(out[0, 0, 0, row, col]) - (local + cross)))
    outputs = {'dv': untile(out)}
    method = 'tril(A)^T @ do + (k * exp2(g_last-g)) @ dh; dh uses original contiguous pointer order'
    assert all((v.dtype == torch.float64 and torch.isfinite(v).all() for v in outputs.values()))
    assert max(scalar_checks) < 1e-09, scalar_checks
    return (outputs, {'method': method, 'scalar_checked_values': len(scalar_checks), 'scalar_max_abs_difference': max(scalar_checks)})
