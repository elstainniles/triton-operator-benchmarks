"""Existing independent CPU equations; file-generation main omitted."""
import math
import torch

def reference(cid, d):
    i = ['cv_e2403d82a21a9bc8', 'cv_f9c1be070ffc815b', 'cv_91595be48834ddae', 'cv_e2cb74da33047e08', 'cv_4320de0cfab1090a', 'cv_ac1f93db9df290f8', 'cv_e2c4857dc1e2e380'].index('cv_e2c4857dc1e2e380')
    kind = 'oja_dv' if i < 2 else 'oja_inter' if i < 4 else 'oja_intra' if i < 6 else 'based'
    if kind == 'based':
        q, k, v = (d[n].double() for n in ('q', 'k', 'v'))
        s = q * d['scale'] @ k.transpose(-1, -2)
        weights = (1 + s + 0.5 * s.square()).tril()
        gold = {'o': weights @ v, 'z': weights.sum(-1)}
        checks = []
        for row, col in [(0, 0), (7, 1), (q.shape[-2] - 1, v.shape[-1] - 1)]:
            scalars = [math.fsum((float(q[0, 0, row, j]) * d['scale'] * float(k[0, 0, t, j]) for j in range(q.shape[-1]))) for t in range(row + 1)]
            weights_s = [1 + s + 0.5 * s * s for s in scalars]
            checks.append(abs(float(gold['o'][0, 0, row, col]) - math.fsum((w * float(v[0, 0, t, col]) for t, w in enumerate(weights_s)))))
            checks.append(abs(float(gold['z'][0, 0, row]) - math.fsum(weights_s)))
        method = 'FP64 causal Taylor attention (1 + s + s^2/2), s=(q*scale)@k^T; full o and z'
    else:
        gold = oja_reference(kind, {'input_data': d})
        checks = []
        bt = d['BT']
        width = d['v'].shape[-1] if kind != 'oja_inter' else d['q'].shape[-1]
        for row, col in [(0, 0), (bt - 1, width - 1)]:
            if kind == 'oja_inter':
                q = (d['q'][0, row, 0].double() * d['scale']).float().double()
                expected = math.fsum((float(q[j]) * float(d['h'][0, 0, 0, j, col]) for j in range(width))) * math.exp(float(d['gv'][0, row, 0, col]))
                checks.append(abs(float(gold['o'][0, row, 0, col]) - expected))
                expected_a = math.fsum((float(q[j]) * float(d['k'][0, 0, 0, j]) for j in range(width)))
                checks.append(abs(float(gold['A'][0, row, 0, 0]) - expected_a))
            elif kind == 'oja_intra':
                expected = float(d['o'][0, row, 0, col]) + math.fsum((float(d['A'][0, row, 0, j]) * math.exp(float(d['gv'][0, row, 0, col]) - float(d['gv'][0, j, 0, col])) * float(d['v'][0, j, 0, col]) for j in range(row + 1)))
                checks.append(abs(float(gold['o'][0, row, 0, col]) - expected))
            else:
                expected = float(d['dv'][0, row, 0, col]) + math.fsum((float(d['A'][0, j, 0, row]) * math.exp(float(d['g'][0, j, 0, col]) - float(d['g'][0, row, 0, col])) * float(d['do'][0, j, 0, col]) for j in range(row, bt)))
                checks.append(abs(float(gold['dv2'][0, row, 0, col]) - expected))
                dg = float(d['o'][0, row, 0, col]) * float(d['do'][0, row, 0, col]) - float(d['v'][0, row, 0, col]) * expected
                checks.append(abs(float(gold['dg'][0, row, 0, col]) - dg))
        method = 'Existing independent CPU64 causal gated OJA mathematics; scalar dot/sum cross-check; original explicit q scale FP32 intermediate retained'
    assert max(checks) < 1e-09, checks
    assert all((t.dtype == torch.float64 and torch.isfinite(t).all() for t in gold.values()))
    details = {'method': method, 'scalar_check_count': len(checks), 'scalar_max_abs_difference': max(checks), 'final_reference_rounding': False}
    return (gold, details)

from original_oja import oja_reference
