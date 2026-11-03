"""Existing independent CPU equations; file-generation main omitted."""
from reference_impl_tile import tile
from reference_impl_scalar_tile import scalar_tile
import math
import torch

def reference(cid, d):
    bt = d['BT']
    errors = []
    q = tile(d['q'], bt).expand(-1, -1, d['H'], -1, -1)
    do = tile(d['do'], bt)
    g = scalar_tile(d['g'], bt)
    level = int(math.log2(bt)) + 1 + d['ell']
    assert d['L'] > level
    l = tile(d['l'], bt)[..., level]
    b, n, h, _, width = q.shape
    vd = d['V']
    flag = 1 << d['ell']
    dh = d['dh'].double().clone()
    dg = d['dg_last'].double().clone()
    state = torch.zeros((b, h, width, vd), dtype=torch.float64)
    hl = d['h_l'].double()
    for chunk in range(n - 1, -1, -1):
        old = dh[:, chunk].clone()
        if not chunk & flag:
            dh[:, chunk] = old + state
            if chunk > 0 and chunk - 1 & flag:
                state = torch.zeros_like(state)
        else:
            dg[:, chunk] += (hl[:, chunk] * (state + old)).sum((-2, -1))
        state = state * torch.exp(g[:, chunk, :, -1])[..., None, None]
        if chunk & flag:
            state = state + (q[:, chunk] * (torch.exp(g[:, chunk]) * l[:, chunk])[..., None]).transpose(-1, -2) @ do[:, chunk]
    out = {'dh': dh, 'dg_last': dg}
    errors.append(abs(float(dg[0, n - 1, 0]) - float(d['dg_last'][0, n - 1, 0]) - math.fsum((float(hl[0, n - 1, 0, ki, vi]) * float(d['dh'][0, n - 1, 0, ki, vi]) for ki in range(width) for vi in range(vd)))))
    method = 'Independent CPU64 reverse binary-level state-gradient accumulation, original in-place dh/dg_last pre-state and valid L8 fixture'
    assert all((v.dtype == torch.float64 and torch.isfinite(v).all() for v in out.values()))
    assert max(errors) < 1e-09, errors
    return (out, {'method': method, 'scalar_checked_values': len(errors), 'scalar_max_abs_difference': max(errors)})
