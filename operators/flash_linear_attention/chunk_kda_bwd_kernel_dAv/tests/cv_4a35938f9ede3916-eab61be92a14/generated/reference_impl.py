"""Existing independent CPU equations; file-generation main omitted."""
from reference_impl_tile import tile
import math
from reference_impl_untile import untile
import torch

def reference(cid, d):
    errors = []
    bt = d['chunk_size']
    v, a, do = (tile(d[n], bt) for n in ['v', 'A', 'do'])
    da = torch.tril(do @ v.transpose(-1, -2)) * d['scale']
    dv = torch.tril(a).transpose(-1, -2) @ do
    for row, col in [(0, 0), (1, 1), (bt - 1, bt - 1)]:
        expected = math.fsum((float(do[0, 0, 0, row, i]) * float(v[0, 0, 0, col, i]) for i in range(v.shape[-1]))) * d['scale']
        errors.append(abs(float(da[0, 0, 0, row, col]) - expected))
    out = {'dA': untile(da), 'dv': untile(dv)}
    method = 'CPU64 dA=tril(do@vT)*scale; dv=tril(A)T@do'
    assert all((x.dtype == torch.float64 and torch.isfinite(x).all() for x in out.values()))
    assert max(errors) < 1e-09, errors
    return (out, {'method': method, 'scalar_checked_values': len(errors), 'scalar_max_abs_difference': max(errors)})
