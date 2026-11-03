"""Existing independent CPU equations; file-generation main omitted."""
import math
import torch

def heads(x):
    return x.double().permute(0, 2, 1, 3)

def unheads(x):
    return x.permute(0, 2, 1, 3).contiguous()

def reference(cid, d):
    checks = []
    notes = {}
    assert d['g_cumsum'] is None and d['dg_cumsum'] is None
    assert d['BV'] >= d['V'] and d['BK'] < d['K']
    q, k = (heads(d['q'][..., :d['BK']]), heads(d['k'][..., :d['BK']]))
    k = k.repeat_interleave(d['G'], dim=1)
    v = heads(d['v']).repeat_interleave(d['G'], dim=1)
    do = heads(d['do'])
    score = q @ k.transpose(-1, -2) * (d['scale'] * 1.4426950216)
    p = torch.exp2(score - d['lse'].double().permute(0, 2, 1)[..., None])
    p = p.tril()
    ds = p * (do @ v.transpose(-1, -2) - d['delta'].double().permute(0, 2, 1)[..., None])
    expected = d['dq'].double().clone()
    expected[..., :d['BK']] = unheads(ds @ k * d['scale'])
    for t, column in [(0, 0), (17, 3), (d['T'] - 1, 15)]:
        val = math.fsum((2.0 ** (math.fsum((float(q[0, 0, t, j]) * float(k[0, 0, s, j]) for j in range(d['BK']))) * d['scale'] * 1.4426950216 - float(d['lse'][0, t, 0])) * (math.fsum((float(do[0, 0, t, j]) * float(v[0, 0, s, j]) for j in range(d['V']))) - float(d['delta'][0, t, 0])) * float(k[0, 0, s, column]) * d['scale'] for s in range(t + 1)))
        checks.append(abs(float(expected[0, t, 0, column]) - val))
    assert torch.equal(expected[..., d['BK']:], d['dq'][..., d['BK']:].double())
    out = {'dq': expected}
    method = 'Independent CPU64 causal softmax dq using original BK=16 prefix; preserve untouched dq columns 16:64'
    notes.update(written_columns=d['BK'], preserved_columns=d['K'] - d['BK'], null_outputs=['dg_cumsum'])
    assert all((x.dtype == torch.float64 and torch.isfinite(x).all() for x in out.values()))
    assert checks and max(checks) < 1e-08, ('cv_028754da723fabbd', checks)
    return (out, {'method': method, 'scalar_checked_values': len(checks), 'scalar_max_abs_difference': max(checks), **notes})
