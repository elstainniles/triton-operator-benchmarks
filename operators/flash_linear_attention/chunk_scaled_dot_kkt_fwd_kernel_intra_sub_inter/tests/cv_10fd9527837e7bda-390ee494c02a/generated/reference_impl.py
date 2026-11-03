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
    k, g = (tile(d['k'], bt), tile(d['g'], bt))
    beta = scalar_tile(d['beta'], bt)
    weights = k * torch.exp(g) @ (k * torch.exp(-g)).transpose(-1, -2) * beta[..., None]
    row = torch.arange(bt)
    written = row[:, None] // d['BC'] > row[None, :] // d['BC']
    out = tile(d['A'], bt).clone()
    out[..., written] = weights[..., written]
    for ri, cj in [(d['BC'], 0), (bt - 1, 0), (bt - 1, bt - d['BC'] - 1)]:
        expected = math.fsum((float(k[0, 0, 0, ri, kd]) * float(k[0, 0, 0, cj, kd]) * math.exp(float(g[0, 0, 0, ri, kd] - g[0, 0, 0, cj, kd])) for kd in range(k.shape[-1]))) * float(beta[0, 0, 0, ri])
        scalar_checks.append(abs(float(out[0, 0, 0, ri, cj]) - expected))
    assert torch.equal(out[..., ~written], tile(d['A'], bt)[..., ~written])
    outputs = {'A': untile(out)}
    method = '(k_i * exp(g_i)) dot (k_j * exp(-g_j)) * beta_i for lower BC blocks; all untouched caller-owned A values preserved'
    assert all((v.dtype == torch.float64 and torch.isfinite(v).all() for v in outputs.values()))
    assert max(scalar_checks) < 1e-09, scalar_checks
    return (outputs, {'method': method, 'scalar_checked_values': len(scalar_checks), 'scalar_max_abs_difference': max(scalar_checks)})
