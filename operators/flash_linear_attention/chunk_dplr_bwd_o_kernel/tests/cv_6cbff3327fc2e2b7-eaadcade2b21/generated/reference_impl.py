"""Existing independent CPU equations; file-generation main omitted."""
from reference_impl_tile import tile
import math
from reference_impl_untile import untile
import torch

def reference(cid, d):
    bt = d['chunk_size']
    checks = []
    k, b, v, vn, do, dv, g = (tile(d[n], bt) for n in ('k', 'b', 'v', 'v_new', 'do', 'dv', 'gk'))
    batch, chunks, heads, _, width = k.shape
    h, dh = (d[n].double().reshape(batch, chunks, heads, width, v.shape[-1]) for n in ('h', 'dh'))
    dq = do @ h.transpose(-1, -2)
    dk = v @ dh.transpose(-1, -2)
    dw = dv @ h.transpose(-1, -2)
    db = vn @ dh.transpose(-1, -2)
    dg = (h * dh).sum(-1) * torch.exp2(g[..., -1, :]) + (k * dk + b * db).sum(-2)
    for row, col in ((0, 0), (bt - 1, width - 1)):
        for output, left, right in ((dq, do, h), (dk, v, dh), (dw, dv, h), (db, vn, dh)):
            scalar = math.fsum((float(left[0, 0, 0, row, j]) * float(right[0, 0, 0, col, j]) for j in range(v.shape[-1])))
            checks.append(abs(float(output[0, 0, 0, row, col]) - scalar))
        scalar = math.fsum((float(h[0, 0, 0, col, j]) * float(dh[0, 0, 0, col, j]) for j in range(v.shape[-1]))) * 2.0 ** float(g[0, 0, 0, -1, col])
        scalar += math.fsum((float(k[0, 0, 0, t, col]) * float(dk[0, 0, 0, t, col]) + float(b[0, 0, 0, t, col]) * float(db[0, 0, 0, t, col]) for t in range(bt)))
        checks.append(abs(float(dg[0, 0, 0, col]) - scalar))
    out = {'dq': untile(dq), 'dk': untile(dk), 'dw': untile(dw), 'db': untile(db), 'dgk_last': dg}
    method = 'Independent CPU64 output/state derivatives and per-chunk gated state contraction; all five original outputs'
    assert all((x.dtype == torch.float64 and torch.isfinite(x).all() for x in out.values()))
    assert checks and max(checks) < 1e-08, checks
    return (out, {'method': method, 'scalar_checked_values': len(checks), 'scalar_max_abs_difference': max(checks), 'all_defined_original_outputs_compared': list(out), 'reference_unrounded': True})
