"""Existing independent CPU equations; file-generation main omitted."""
from reference_impl_tile import tile
import math
from reference_impl_untile import untile
import torch

def reference(cid, d):
    errors = []
    bt = d['chunk_size']
    ag, v, a, c = (tile(d[n], bt) for n in ['ag', 'v', 'A_ab_inv', 'A_ak'])
    a = torch.tril(a)
    c = torch.tril(c, diagonal=-1)
    w = a @ ag
    u = a @ c @ v
    for row, col in [(0, 0), (1, 1), (bt - 1, ag.shape[-1] - 1)]:
        expected = math.fsum((float(a[0, 0, 0, row, j]) * float(ag[0, 0, 0, j, col]) for j in range(bt)))
        errors.append(abs(float(w[0, 0, 0, row, col]) - expected))
    out = {'w': untile(w), 'u': untile(u)}
    method = 'CPU64 w=tril(A_ab_inv)@ag and u=tril(A_ab_inv)@strict_tril(A_ak)@v'
    assert all((x.dtype == torch.float64 and torch.isfinite(x).all() for x in out.values()))
    assert max(errors) < 1e-08, errors
    return (out, {'method': method, 'scalar_checked_values': len(errors), 'scalar_max_abs_difference': max(errors)})
