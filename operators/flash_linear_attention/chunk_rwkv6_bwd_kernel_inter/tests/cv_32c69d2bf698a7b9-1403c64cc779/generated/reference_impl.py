"""Existing independent CPU equations; file-generation main omitted."""
from reference_impl_tile import tile
import math
from reference_impl_untile import untile
import torch

def reference(cid, d):
    bt = d.get('BT', d.get('chunk_size'))
    errors = []
    q, k, v, gi, ge, do, da, dq0, dk0 = (tile(d[n], bt) for n in ['q', 'k', 'v', 'gi', 'ge', 'do', 'dA', 'dq', 'dk'])
    b, n, h, _, width = q.shape
    vd = v.shape[-1]
    state, ds = (d[nm].double().reshape(b, n, h, width, vd) for nm in ['h', 'dh'])
    dgk = (state * ds).sum(-1) * torch.exp(gi[..., -1, :])
    dq = do @ state.transpose(-1, -2) * d['scale'] * torch.exp(ge)
    dk = v @ ds.transpose(-1, -2) * torch.exp(gi[..., -1:, :] - gi)
    dgk = dgk + (dk * k).sum(-2)
    dq = dq + dq0
    dk = dk + dk0
    difference = q * dq - k * dk
    reverse = torch.flip(torch.flip(difference, [-2]).cumsum(-2), [-2])
    dg = reverse + dgk[..., None, :] - q * dq
    diagonal = da.diagonal(dim1=-2, dim2=-1)
    u = d['u'].double()[None, None, :, None, :]
    dq2 = dq + diagonal[..., None] * u * k
    dk2 = dk + diagonal[..., None] * u * q
    du = (diagonal[..., None] * q * k).sum(-2)
    for row, col in [(0, 0), (1, 1), (bt - 1, width - 1)]:
        query = math.fsum((float(do[0, 0, 0, row, i]) * float(state[0, 0, 0, col, i]) for i in range(vd))) * d['scale'] * math.exp(float(ge[0, 0, 0, row, col])) + float(dq0[0, 0, 0, row, col])
        expected = query + float(diagonal[0, 0, 0, row]) * float(d['u'][0, col]) * float(k[0, 0, 0, row, col])
        errors.append(abs(float(dq2[0, 0, 0, row, col]) - expected))
        expected_u = math.fsum((float(da[0, 0, 0, t, t]) * float(q[0, 0, 0, t, col]) * float(k[0, 0, 0, t, col]) for t in range(bt)))
        errors.append(abs(float(du[0, 0, 0, col]) - expected_u))
    out = {'dq2': untile(dq2), 'dk2': untile(dk2), 'dg': untile(dg), 'du': du.reshape_as(d['du'])}
    method = 'Independent CPU64 RWKV inter-state matrix derivatives, reverse gate sum, diagonal-u corrections and per-chunk du reduction'
    assert all((v.dtype == torch.float64 and torch.isfinite(v).all() for v in out.values()))
    assert max(errors) < 1e-09, errors
    return (out, {'method': method, 'scalar_checked_values': len(errors), 'scalar_max_abs_difference': max(errors)})
