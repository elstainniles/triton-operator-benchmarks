"""Existing independent CPU equations; file-generation main omitted."""
from reference_impl_tile import tile
from reference_impl_scalar_tile import scalar_tile
import math
from reference_impl_untile import untile
import torch

def reference(cid, d):
    errors = []
    bt = d['BT']
    q, k, v = (tile(d[n], bt) for n in ['q', 'k', 'v'])
    g, beta = (scalar_tile(d[n], bt) for n in ['g', 'beta'])
    batch, nt, heads, _, width = q.shape
    h = d['h'].double().reshape(batch, nt, heads, width, width)
    hkv = d['h_kv'].double().reshape(batch, nt, heads, width, width)
    weight = beta * torch.exp(-g)
    outer = k[..., :, None] * k[..., None, :] * weight[..., None, None]
    systems = (outer.cumsum(-3) + h[..., None, :, :]) * torch.exp(g)[..., None, None]
    systems = systems + torch.diag_embed(d['lamb'].double())[None, None, :, None, :, :]
    x = torch.zeros_like(q)
    residual = q.clone()
    direction = q.clone()
    delta = residual.square().sum(-1)
    for _ in range(d['max_CG_iteration']):
        product = (direction[..., None, :] @ systems).squeeze(-2)
        alpha = delta / ((direction * product).sum(-1) + 1e-05)
        x = x + alpha[..., None] * direction
        residual = residual - alpha[..., None] * product
        new_delta = residual.square().sum(-1)
        direction = residual + (new_delta / (delta + 1e-05))[..., None] * direction
        delta = new_delta
    cross = ((k[..., :, None] * v[..., None, :] * weight[..., None, None]).cumsum(-3) + hkv[..., None, :, :]) * torch.exp(g)[..., None, None]
    output = (x[..., None, :] @ cross).squeeze(-2)
    for row, col in [(0, 0), (1, 1), (bt - 1, width - 1)]:
        expected = math.fsum((float(x[0, 0, 0, row, j]) * float(hkv[0, 0, 0, j, col]) for j in range(width))) * math.exp(float(g[0, 0, 0, row]))
        expected += math.fsum((math.fsum((float(x[0, 0, 0, row, j]) * float(k[0, 0, 0, t, j]) for j in range(width))) * float(v[0, 0, 0, t, col]) * float(beta[0, 0, 0, t]) * math.exp(float(g[0, 0, 0, row] - g[0, 0, 0, t])) for t in range(row + 1)))
        errors.append(abs(float(output[0, 0, 0, row, col]) - expected))
    out = {'q_final': untile(x), 'o': untile(output)}
    method = 'Independent CPU64 per-token dense covariance systems; original 30 CG iterations and 1e-5 denominators, followed by dense key-value projection'
    assert all((x.dtype == torch.float64 and torch.isfinite(x).all() for x in out.values()))
    assert max(errors) < 1e-09, errors
    return (out, {'method': method, 'scalar_checked_values': len(errors), 'scalar_max_abs_difference': max(errors)})
