"""Existing independent CPU equations; file-generation main omitted."""
from reference_impl_tile import tile
from reference_impl_scalar_tile import scalar_tile
import math
from reference_impl_untile import untile
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
    k = tile(d['k'], bt).expand(-1, -1, h, -1, -1)
    v = tile(d['v'], bt)
    dq = tile(d['dq'], bt).clone()
    dg = scalar_tile(d['dg'], bt).clone()
    dl = tile(d['dl'], bt).clone()
    hl = d['h_l'].double().clone()
    state = torch.zeros((b, h, width, vd), dtype=torch.float64)
    for chunk in range(n):
        if chunk & flag:
            deriv = do[:, chunk] @ state.transpose(-1, -2) * torch.exp(g[:, chunk])[..., None]
            dlvalue = (deriv * q[:, chunk]).sum(-1)
            hl[:, chunk] = state
            dq[:, chunk] += deriv * l[:, chunk, ..., None]
            dl[:, chunk, ..., level] = dlvalue
            dg[:, chunk] += dlvalue * l[:, chunk]
            if not chunk + 1 & flag:
                state = torch.zeros_like(state)
        state = state * torch.exp(g[:, chunk, :, -1])[..., None, None]
        if not chunk & flag:
            state = state + (k[:, chunk] * torch.exp(g[:, chunk, :, -1, None] - g[:, chunk])[..., None]).transpose(-1, -2) @ v[:, chunk]
    out = {'h_l': hl, 'dq': untile(dq), 'dg': untile(dg[..., None]).squeeze(-1), 'dl': untile(dl)}
    chunk = flag
    for row, col in [(0, 0), (1, 1), (bt - 1, width - 1)]:
        deriv = math.fsum((float(do[0, chunk, 0, row, i]) * float(hl[0, chunk, 0, col, i]) for i in range(vd))) * math.exp(float(g[0, chunk, 0, row]))
        expected = float(d['dq'][0, chunk * bt + row, 0, col]) + deriv * float(l[0, chunk, 0, row])
        errors.append(abs(float(dq[0, chunk, 0, row, col]) - expected))
    method = 'Independent CPU64 forward binary-level state accumulation and analytic q/g/level derivatives; preserves original untouched outputs'
    assert all((v.dtype == torch.float64 and torch.isfinite(v).all() for v in out.values()))
    assert max(errors) < 1e-09, errors
    return (out, {'method': method, 'scalar_checked_values': len(errors), 'scalar_max_abs_difference': max(errors)})
