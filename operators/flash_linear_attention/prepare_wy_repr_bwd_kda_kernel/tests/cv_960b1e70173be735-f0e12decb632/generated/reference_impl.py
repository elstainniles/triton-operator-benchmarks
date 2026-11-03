"""Existing independent CPU equations; file-generation main omitted."""
from reference_impl_tile import tile
from reference_impl_scalar_tile import scalar_tile
import math
from reference_impl_untile import untile
import torch

def reference(cid, d):
    errors = []
    bt = d['A'].shape[-1]
    k, v, gk, A, dw, du, dk0, dg0 = (tile(d[n], bt) for n in ['k', 'v', 'gk', 'A', 'dw', 'du', 'dk', 'dg'])
    beta = scalar_tile(d['beta'], bt)
    assert k.shape[2] == v.shape[2]
    e = torch.exp2(gk)
    at = A.transpose(-1, -2)
    kbg = k * beta[..., None] * e
    grad_kbg = at @ dw
    grad_vb = at @ du
    dk = dk0 + grad_kbg * e * beta[..., None]
    dg = dg0 + kbg * grad_kbg
    dv = grad_vb * beta[..., None]
    db = (grad_kbg * k * e).sum(-1) + (grad_vb * v).sum(-1)
    da = dw @ kbg.transpose(-1, -2) + du @ (v * beta[..., None]).transpose(-1, -2)
    da = -torch.tril(at @ torch.tril(da, diagonal=-1) @ at, diagonal=-1)
    for ri, ci in [(0, 0), (bt - 1, 1), (bt // 2, k.shape[-1] - 1)]:
        grad = math.fsum((float(A[0, 0, 0, j, ri]) * float(dw[0, 0, 0, j, ci]) for j in range(bt)))
        expected = float(dk0[0, 0, 0, ri, ci]) + grad * 2.0 ** float(gk[0, 0, 0, ri, ci]) * float(beta[0, 0, 0, ri])
        errors.append(abs(float(dk[0, 0, 0, ri, ci]) - expected))
    out = {n: untile(x) for n, x in [('dk_out', dk), ('dv', dv), ('dg_out', dg), ('dA', da)]}
    out['db'] = untile(db[..., None]).squeeze(-1)
    method = 'CPU64 KDA WY adjoint, lower-triangular inverse derivative and caller dk/dg accumulation; original A and input states retained'
    assert all((x.dtype == torch.float64 and torch.isfinite(x).all() for x in out.values()))
    assert max(errors) < 1e-08, errors
    return (out, {'method': method, 'independent_cross_checks': len(errors), 'maximum_cross_check_error': max(errors)})
