"""Existing independent CPU equations; file-generation main omitted."""
import torch

def reference(d):
    q, k, u, dv = (d[name].double().transpose(0, 1) for name in ('q', 'k', 'u', 'grad_v'))
    params = d['params']
    rowdot = d['row_dot_sum'].to(d['k'].dtype).double().transpose(0, 1)
    prob = torch.exp2(q @ k.transpose(-1, -2) * params['fa_scale'] - d['lse'].double().transpose(0, 1)[..., None]) * d['beta'].double().transpose(0, 1)[..., None]
    da = torch.tril(prob * (-dv @ u.transpose(-1, -2) - rowdot[..., None]), diagonal=-1)
    da = da.to(d['q'].dtype).double()
    dq = da @ k * params['qk_scale']
    dk = da.transpose(-1, -2) @ q * params['qk_scale']
    qv = q[0].clone().requires_grad_(True)
    kv = k[0].clone().requires_grad_(True)
    score = qv @ kv.T * params['qk_scale']
    objective = (score * da[0]).sum()
    aq, ak = torch.autograd.grad(objective, (qv, kv))
    torch.testing.assert_close(aq, dq[0], rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(ak, dk[0], rtol=1e-11, atol=1e-11)
    for row, col in [(0, 0), (7, 3), (q.shape[-2] - 1, q.shape[-1] - 1)]:
        import math
        scalar = math.fsum((float(da[0, row, j]) * float(k[0, j, col]) for j in range(row))) * params['qk_scale']
        assert abs(float(dq[0, row, col]) - scalar) < 1e-09
    gold = {'grad_q': dq.transpose(0, 1).contiguous(), 'grad_k': dk.transpose(0, 1).contiguous()}
    assert all((torch.isfinite(v).all() and v.dtype == torch.float64 for v in gold.values()))
    return (gold, {'method': 'Strict causal exp2 attention Jacobian with supplied lse/beta/row_dot; dq=da@k and dk=da.T@q; preserve explicit row_dot and da casts', 'head0_full_bilinear_autograd_crosscheck': True, 'scalar_dq_checks': 3, 'reference_final_rounding': False})
