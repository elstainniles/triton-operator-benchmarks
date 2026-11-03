"""Existing independent CPU equations; file-generation main omitted."""
import math
import torch

def reference(cid, d):
    bt = d.get('BT', d.get('chunk_size'))
    errors = []
    b, h, t, kdim = d['q'].shape
    assert t % bt == 0

    def head_tile(x):
        return x.double().reshape(b, h, t // bt, bt, x.shape[-1])
    q, k, v, A = (head_tile(d[n]) for n in ['q', 'k', 'v', 'A'])
    q = q * d['scale']
    beta = d['beta'].double().reshape(b, h, t // bt, bt)
    qkt = torch.tril(q @ k.transpose(-1, -2)) @ A
    kkt = torch.tril(k @ k.transpose(-1, -2), diagonal=-1) @ A
    kb = k * beta[..., None]
    qn = q - qkt @ kb
    kn = k - kkt.transpose(-1, -2) @ kb
    o = qkt @ v
    for row, col in [(0, 0), (1, 1), (bt - 1, v.shape[-1] - 1)]:
        expected = math.fsum((float(qkt[0, 0, 0, row, j]) * float(v[0, 0, 0, j, col]) for j in range(bt)))
        errors.append(abs(float(o[0, 0, 0, row, col]) - expected))
    out = {'q_new': qn.reshape_as(d['q']), 'k_new': kn.reshape_as(d['k']), 'o': o.reshape_as(d['v'])}
    if d['output_attentions']:
        out['A_local'] = qkt.reshape_as(d['A'])
    method = 'CPU64 head-first triangular q/k Gram matrices, original A transform, q_new/k_new and local output'
    assert all((v.dtype == torch.float64 and torch.isfinite(v).all() for v in out.values()))
    assert max(errors) < 1e-08, errors
    return (out, {'method': method, 'scalar_checked_values': len(errors), 'scalar_max_abs_difference': max(errors)})
