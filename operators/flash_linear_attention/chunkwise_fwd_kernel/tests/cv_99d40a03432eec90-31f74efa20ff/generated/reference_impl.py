"""Existing independent CPU equations; file-generation main omitted."""
import math
import torch

def reference(cid, d):
    bt = d['BT']
    errors = []
    assert d['h0'] is None and d['ht'] is None and (d['offsets'] is None)
    q, k = (d['q'].double().squeeze(2), d['k'].double().squeeze(2))
    v = d['v'].double().transpose(1, 2)
    g = d['g'].double().transpose(1, 2)
    b, h, t, _ = v.shape
    nt = t // bt
    blocks = g.reshape(b, h, nt, bt)
    offsets = torch.cat((torch.zeros_like(blocks[..., :1, -1]), blocks[..., :-1, -1].cumsum(-1)), dim=-1)
    cumulative = (blocks + offsets[..., None]).reshape(b, h, t)
    levels = torch.zeros((t, t), dtype=torch.long)
    allowed = torch.zeros((t, t), dtype=torch.bool)
    for i in range(t):
        for j in range(i + 1):
            if i // bt == j // bt:
                level = (i % bt ^ j % bt).bit_length()
                valid = True
            else:
                ell = (i // bt ^ j // bt).bit_length() - 1
                level = int(math.log2(bt)) + 1 + ell
                valid = ell <= d['MAX_LEVEL']
            levels[i, j] = level
            allowed[i, j] = valid
    scales = d['level_scales'].double().transpose(1, 2)
    weights = scales.gather(-1, levels[None, None].expand(b, h, t, t)) * torch.exp(cumulative[..., None] - cumulative[..., None, :])
    weights = weights.masked_fill(~allowed, 0)
    parts = []
    for ki in range(0, d['K'], 32):
        score = q[:, :, ki:ki + 32] @ k[:, :, ki:ki + 32].transpose(-1, -2)
        parts.append(score[:, None] * weights @ v)
    out_value = torch.stack(parts, dim=-2).transpose(1, 2)
    for row, col in [(0, 0), (bt, 1), (t - 1, d['V'] - 1)]:
        expected = math.fsum((math.fsum((float(q[0, row, ki]) * float(k[0, j, ki]) for ki in range(32))) * float(weights[0, 0, row, j]) * float(v[0, 0, j, col]) for j in range(row + 1)))
        errors.append(abs(float(out_value[0, row, 0, 0, col]) - expected))
    out = {'o': out_value, 'new_offsets': torch.zeros(b, dtype=torch.float64)}
    method = 'Independent CPU64 dense binary-level causal attention, globalized per-chunk gate sums and separate original BK32 output partials; no final-state write keeps new_offsets zero'
    assert all((v.dtype == torch.float64 and torch.isfinite(v).all() for v in out.values()))
    assert max(errors) < 1e-09, errors
    return (out, {'method': method, 'scalar_checked_values': len(errors), 'scalar_max_abs_difference': max(errors)})
