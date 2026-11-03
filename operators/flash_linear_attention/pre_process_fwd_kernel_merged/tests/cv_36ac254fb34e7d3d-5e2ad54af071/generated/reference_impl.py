"""Existing independent CPU equations; file-generation main omitted."""
import torch

def reference(cid, d):
    checks = []
    notes = {}
    assert all((d[n] is None for n in ('g', 'gk', 'bg', 'v', 'cu_seqlens')))
    assert d['k'].shape[0] == 1
    k, w, u = (d[n].double()[0].permute(1, 0, 2) for n in ('k', 'w', 'u'))
    hv, length, width = w.shape
    k = k.repeat_interleave(hv // k.shape[0], dim=0)
    value = u.shape[-1]
    h = torch.zeros(hv, width, value, dtype=torch.float64)
    m = torch.eye(width, dtype=torch.float64).expand(hv, width, width).clone()
    affine_h = torch.zeros_like(h)
    identity = torch.eye(width, dtype=torch.float64)
    for begin in range(0, length, d['chunk_size']):
        kt, wt, ut = (x[:, begin:begin + d['chunk_size']] for x in (k, w, u))
        h = h + kt.transpose(-1, -2) @ (ut - wt @ h)
        transform = identity - kt.transpose(-1, -2) @ wt
        m = transform @ m
        affine_h = transform @ affine_h + kt.transpose(-1, -2) @ ut
    checks.append(float((h - affine_h).abs().max()))
    out = {'hm': torch.cat((h, m), dim=-1)}
    method = 'Independent CPU64 affine chunk recurrence h <- h + k^T(u-w h), M <- (I-k^T w) M; grouped heads and original MULTI_SEQS=False preserved'
    notes.update(original_absolute_tolerance=1e-05, actual_output_dtype='float32', independently_checked_affine_recurrence=True)
    assert all((x.dtype == torch.float64 and torch.isfinite(x).all() for x in out.values()))
    assert checks and max(checks) < 1e-08, ('cv_36ac254fb34e7d3d', checks)
    return (out, {'method': method, 'scalar_checked_values': len(checks), 'scalar_max_abs_difference': max(checks), **notes})
