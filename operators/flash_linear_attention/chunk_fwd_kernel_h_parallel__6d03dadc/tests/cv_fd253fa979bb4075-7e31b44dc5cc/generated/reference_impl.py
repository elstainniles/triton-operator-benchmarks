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
    out = {'h': torch.cat([initial[:, None], updates[:, :-1]], dim=1)}
    if d['output_final_state']:
        out['ht'] = updates[:, -1]
    method = 'Independent CPU64 per-chunk k^T@v; original parallel kernel writes initial h0 and each chunk contribution into next slot'
    for kc, vc in [(0, 0), (1, 1), (width - 1, value_width - 1)]:
        expected = math.fsum((float(k[0, 0, 0, t, kc]) * float(v[0, 0, 0, t, vc]) for t in range(bt)))
        errors.append(abs(float(updates[0, 0, 0, kc, vc]) - expected))
    assert all((v.dtype == torch.float64 and torch.isfinite(v).all() for v in out.values()))
    assert max(errors) < 1e-09, errors
    return (out, {'method': method, 'scalar_checked_values': len(errors), 'scalar_max_abs_difference': max(errors)})
