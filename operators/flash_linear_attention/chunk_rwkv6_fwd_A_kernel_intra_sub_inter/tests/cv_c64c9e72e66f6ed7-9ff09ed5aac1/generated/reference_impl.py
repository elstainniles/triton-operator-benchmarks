"""Existing independent CPU equations; file-generation main omitted."""
from reference_impl_tile import tile
import math
from reference_impl_untile import untile
import torch

def reference(cid, d):
    errors = []
    bt = d['chunk_size']
    q, k, gi, ge = (tile(d[n], bt) for n in ['q', 'k', 'gi', 'ge'])
    scores = q * torch.exp2(ge) @ (k * torch.exp2(-gi)).transpose(-1, -2) * d['scale']
    row = torch.arange(bt)
    written = row[:, None] // 16 > row[None, :] // 16
    value = torch.zeros_like(scores)
    value[..., written] = scores[..., written]
    for ri, cj in [(16, 0), (bt - 1, 0), (bt - 1, bt - 17)]:
        expected = math.fsum((float(q[0, 0, 0, ri, j]) * float(k[0, 0, 0, cj, j]) * 2.0 ** float(ge[0, 0, 0, ri, j] - gi[0, 0, 0, cj, j]) for j in range(q.shape[-1]))) * d['scale']
        errors.append(abs(float(value[0, 0, 0, ri, cj]) - expected))
    out = {'A': untile(value)}
    method = 'CPU64 RWKV exp2-gated off-diagonal 16x16 blocks; output allocation deterministically initializes untouched cells to zero, original kernel/masks and float32 1e-5 acceptance retained'
    assert all((v.dtype == torch.float64 and torch.isfinite(v).all() for v in out.values()))
    assert max(errors) < 1e-08, errors
    return (out, {'method': method, 'scalar_checked_values': len(errors), 'scalar_max_abs_difference': max(errors)})
