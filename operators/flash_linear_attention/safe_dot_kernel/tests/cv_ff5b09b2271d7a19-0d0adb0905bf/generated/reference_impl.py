"""Existing independent CPU equations; file-generation main omitted."""
import math
import torch

def reference(cid, d):
    errors = []
    a, b = (d['a'].double(), d['b'].double())
    value = a @ b
    for row, col in [(0, 0), (1, 1), (a.shape[0] - 1, b.shape[1] - 1)]:
        expected = math.fsum((float(a[row, j]) * float(b[j, col]) for j in range(a.shape[1])))
        errors.append(abs(float(value[row, col]) - expected))
    out = {'c': value}
    method = 'Independent CPU64 dense matrix product'
    assert all((x.dtype == torch.float64 and torch.isfinite(x).all() for x in out.values()))
    assert max(errors) < 1e-08, errors
    return (out, {'method': method, 'scalar_checked_values': len(errors), 'scalar_max_abs_difference': max(errors)})
