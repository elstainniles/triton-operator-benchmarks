"""Existing independent CPU equations; file-generation main omitted."""
import math
import torch

def reference(cid, d):
    errors = []
    q = d['q'].double()
    b, t, nblocks, hq, k = q.shape
    h = d['k'].shape[2]
    bt, s = (d['BT'], d['S'])
    ns = math.ceil(t / s)
    key = d['k'].double().repeat_interleave(hq // h, dim=2).transpose(1, 2)
    value = d['v'].double().repeat_interleave(hq // h, dim=2).transpose(1, 2)
    do = d['do'].double().transpose(1, 2)
    g = d['g_cumsum'].double().transpose(1, 2)
    L = d['L'].double().transpose(1, 2)
    delta = d['D'].double().transpose(1, 2)
    hc = d['hc_whole'].double().reshape(b, ns, h, k, k).repeat_interleave(hq // h, dim=2)
    dq = torch.zeros(b, hq, t, k, dtype=torch.float64)
    dhc = torch.zeros(b, ns, hq, k, k, dtype=torch.float64)
    dg = d['dg_cumsum'].double().transpose(1, 2).clone()
    for start in range(0, t, bt):
        carry = torch.zeros(b, hq, bt, k, dtype=torch.float64)
        for j in range(start // s):
            query = q[:, start:start + bt, j + 1].transpose(1, 2)
            dhc[:, j] -= query.transpose(-1, -2) @ carry
            carry = carry @ (torch.eye(k, dtype=torch.float64) - hc[:, j].transpose(-1, -2))
            lo, hi = (j * s, (j + 1) * s)
            score = query @ key[:, :, lo:hi].transpose(-1, -2) + g[:, :, start:start + bt, None] - g[:, :, None, lo:hi]
            probability = torch.exp2(score * d['scale'] * 1.44269504 - L[:, :, start:start + bt, None])
            dp = do[:, :, start:start + bt] @ value[:, :, lo:hi].transpose(-1, -2)
            da = probability * (dp - delta[:, :, start:start + bt, None]) * d['scale']
            carry += da @ key[:, :, lo:hi]
            dg[:, :, start:start + bt] += da.sum(-1)
        dq[:, :, start:start + bt] = carry
    row = s
    hi = s
    query = q[0, row, 1, 0]
    for col in [0, k - 1]:
        terms = []
        for j in range(hi):
            score = math.fsum((float(query[z]) * float(key[0, 0, j, z]) for z in range(k))) + float(g[0, 0, row] - g[0, 0, j])
            p = 2.0 ** (score * d['scale'] * 1.44269504 - float(L[0, 0, row]))
            dp = math.fsum((float(do[0, 0, row, z]) * float(value[0, 0, j, z]) for z in range(value.shape[-1])))
            terms.append(p * (dp - float(delta[0, 0, row])) * d['scale'] * float(key[0, 0, j, col]))
        errors.append(abs(float(dq[0, 0, row, col]) - math.fsum(terms)))
    out = {'dq': dq.transpose(1, 2), 'dhc_whole': dhc.reshape(b * ns, hq, k, k), 'dg_cumsum': dg.transpose(1, 2)}
    method = 'CPU64 full-S attention gradient and residual-state adjoint, deterministic sum of all contributions to original atomic_add dhc/dg, original q block selection preserved'
    metadata = {'atomic_output_arguments': [10, 11], 'original_outputs_arguments': {'dq': 8, 'dhc_whole': 10, 'dg_cumsum': 11}}
    assert all((x.dtype == torch.float64 and torch.isfinite(x).all() for x in out.values()))
    assert max(errors) < 1e-08, errors
    return (out, {'method': method, 'scalar_cross_checks': len(errors), 'scalar_max_abs_error': max(errors), **metadata})
