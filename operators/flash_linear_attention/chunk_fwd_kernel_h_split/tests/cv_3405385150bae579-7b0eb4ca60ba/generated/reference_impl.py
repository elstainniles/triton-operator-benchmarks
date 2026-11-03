"""Existing independent CPU equations; file-generation main omitted."""
from reference_impl_tile import tile
import math
import torch

def reference(cid, d):
    bt = d['chunk_size']
    errors = []
    assert d['g'] is None and d['gk'] is None and (d['gv'] is None)
    k, v = (tile(d['k'], bt), tile(d['v'], bt))
    b, n, h, _, width = k.shape
    value_width = v.shape[-1]
    updates = k.transpose(-1, -2) @ v
    initial = torch.zeros((b, h, width, value_width), dtype=torch.float64) if d['h0'] is None else d['h0'].double()
    per_split = d['split_size'] // bt
    assert n % per_split == 0
    summed = updates.reshape(b, n // per_split, per_split, h, width, value_width).sum(2)
    starts = torch.cat([initial[:, None], initial[:, None] + summed.cumsum(1)[:, :-1]], dim=1)
    out = {'hr': starts.reshape(-1, h, width, value_width)}
    if d['output_final_state']:
        out['ht'] = initial + summed.sum(1)
    method = 'Independent CPU64 split sums and exclusive prefix of k^T@v; full original split/reduction wrapper outputs'
    for kc, vc in [(0, 0), (1, 1), (width - 1, value_width - 1)]:
        expected = math.fsum((float(k[0, 0, 0, t, kc]) * float(v[0, 0, 0, t, vc]) for t in range(bt)))
        errors.append(abs(float(updates[0, 0, 0, kc, vc]) - expected))
    assert all((v.dtype == torch.float64 and torch.isfinite(v).all() for v in out.values()))
    assert max(errors) < 1e-09, errors
    return (out, {'method': method, 'scalar_checked_values': len(errors), 'scalar_max_abs_difference': max(errors)})
