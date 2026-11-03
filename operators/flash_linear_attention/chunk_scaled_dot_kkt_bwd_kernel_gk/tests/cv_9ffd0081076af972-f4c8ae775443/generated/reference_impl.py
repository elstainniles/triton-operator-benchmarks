"""Existing independent CPU equations; file-generation main omitted."""
from reference_impl_tile import tile
from reference_impl_scalar_tile import scalar_tile
import math
from reference_impl_untile import untile
import torch

def reference(cid, d):
    bt = d.get('BT', d.get('chunk_size'))
    errors = []
    k = tile(d['k'], bt).requires_grad_()
    g = tile(d['g'], bt).requires_grad_()
    beta = scalar_tile(d['beta'], bt).requires_grad_()
    da = tile(d['dA'], bt)
    a = torch.tril(k * torch.exp(g) @ (k * torch.exp(-g)).transpose(-1, -2) * beta[..., None])
    dk, dg, db = torch.autograd.grad((a * da).sum(), (k, g, beta))
    assert d['db'].shape[0] == 1 and d['BK'] >= d['K']
    for row in [0, 1, bt - 1]:
        expected = math.fsum((float(da[0, 0, 0, row, j]) * math.fsum((float(k[0, 0, 0, row, c]) * float(k[0, 0, 0, j, c]) * math.exp(float(g[0, 0, 0, row, c] - g[0, 0, 0, j, c])) for c in range(k.shape[-1]))) for j in range(row + 1)))
        errors.append(abs(float(db[0, 0, 0, row]) - expected))
    out = {'dk': untile(dk.detach()), 'dg': untile(dg.detach()), 'db': untile(db.detach().unsqueeze(-1)).squeeze(-1).unsqueeze(0)}
    method = 'Independent CPU64 autograd of sum(tril(beta_i * sum_c k_ic*k_jc*exp(g_ic-g_jc))*dA); all original db partials (NK=1)'
    assert all((v.dtype == torch.float64 and torch.isfinite(v).all() for v in out.values()))
    assert max(errors) < 1e-08, errors
    return (out, {'method': method, 'scalar_checked_values': len(errors), 'scalar_max_abs_difference': max(errors)})
