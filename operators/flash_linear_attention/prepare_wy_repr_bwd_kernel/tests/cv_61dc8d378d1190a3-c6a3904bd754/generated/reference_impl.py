"""Existing independent CPU equations; file-generation main omitted."""
from reference_impl_tile import tile
from reference_impl_scalar_tile import scalar_tile
import math
from reference_impl_untile import untile
import torch

def reference(cid, d):
    errors = []
    bt = d['BT']
    k, p, v, A, dw, du = (tile(d[n], bt) for n in ['k', 'p', 'v', 'A', 'dw', 'du'])
    beta, g0, g = (scalar_tile(d[n], bt) for n in ['beta', 'g0', 'g'])
    eg0 = torch.exp2(g0)
    pb = p * beta[..., None] * eg0[..., None]
    vp = v * beta[..., None]
    at = A.transpose(-1, -2)
    grad_p_scaled = at @ dw
    grad_v_scaled = at @ du
    direct_dp = grad_p_scaled * beta[..., None] * eg0[..., None]
    dv = grad_v_scaled * beta[..., None]
    da = dw @ pb.transpose(-1, -2) + du @ vp.transpose(-1, -2)
    da = -torch.tril(at @ torch.tril(da, diagonal=-1) @ at, diagonal=-1) * torch.exp2(g0[..., None] - g[..., None, :])
    dp_scaled = da @ k
    dp = direct_dp + dp_scaled * beta[..., None]
    dk = da.transpose(-1, -2) @ (p * beta[..., None])
    db = (grad_p_scaled * p * eg0[..., None]).sum(-1) + (grad_v_scaled * v).sum(-1) + (dp_scaled * p).sum(-1)
    weighted = da @ k
    da_a = da * (p * beta[..., None] @ k.transpose(-1, -2))
    dg0 = (direct_dp * p).sum(-1) + da_a.sum(-1)
    dg = -da_a.sum(-2)
    for ri, ci in [(0, 0), (bt - 1, 1), (bt // 2, k.shape[-1] - 1)]:
        expected = math.fsum((float(da[0, 0, 0, t, ri]) * float(p[0, 0, 0, t, ci]) * float(beta[0, 0, 0, t]) for t in range(bt)))
        errors.append(abs(float(dk[0, 0, 0, ri, ci]) - expected))
    out = {n: untile(x) for n, x in [('dk', dk), ('dv', dv), ('dp', dp)]}
    out.update({n: untile(x[..., None]).squeeze(-1) for n, x in [('dbeta', db), ('dg0', dg0), ('dg', dg)]})
    method = 'CPU64 adjoint of WY recomputation and lower-triangular inverse, full original six output buffers; BK32 original tile retained'
    assert all((x.dtype == torch.float64 and torch.isfinite(x).all() for x in out.values()))
    assert max(errors) < 1e-08, errors
    return (out, {'method': method, 'independent_cross_checks': len(errors), 'maximum_cross_check_error': max(errors)})
