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

def scalar_tile(t, bt):
    return tile(t.unsqueeze(-1), bt).squeeze(-1)

def reference(cid, d):
    bt = d.get('BT', d.get('chunk_size'))
    scalar_checks = []
    k, p = (tile(d['k'], bt), tile(d['p'], bt))
    beta = scalar_tile(d['beta'], bt)
    out = p * beta[..., None] @ k.transpose(-1, -2)
    if d['g'] is not None:
        g, g0 = (scalar_tile(d['g'], bt), scalar_tile(d['g0'], bt))
        out = out * torch.exp2(g0[..., None] - g[..., None, :])
    out = torch.tril(out, diagonal=-1)
    for row, col in [(1, 0), (bt - 1, 0), (bt - 1, bt - 2)]:
        expected = math.fsum((float(p[0, 0, 0, row, kd]) * float(k[0, 0, 0, col, kd]) for kd in range(k.shape[-1]))) * float(beta[0, 0, 0, row])
        if d['g'] is not None:
            expected *= 2.0 ** float(g0[0, 0, 0, row] - g[0, 0, 0, col])
        scalar_checks.append(abs(float(out[0, 0, 0, row, col]) - expected))
    outputs = {'A': untile(out)}
    method = 'Strictly lower (p * beta) @ k^T with exp2(g0_i-g_j) gate'
    assert all((v.dtype == torch.float64 and torch.isfinite(v).all() for v in outputs.values()))
    assert max(scalar_checks) < 1e-09, scalar_checks
    return (outputs, {'method': method, 'scalar_checked_values': len(scalar_checks), 'scalar_max_abs_difference': max(scalar_checks)})
