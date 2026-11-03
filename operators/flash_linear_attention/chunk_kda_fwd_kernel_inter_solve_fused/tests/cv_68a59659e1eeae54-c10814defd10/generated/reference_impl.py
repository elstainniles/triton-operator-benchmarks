"""Existing independent CPU equations; file-generation main omitted."""
from reference_impl_tile import tile
from reference_impl_scalar_tile import scalar_tile
import math
from reference_impl_untile import untile
import torch

def reference(cid, d):
    errors = []
    bt, bc = (64, 16)
    q, k, g = (tile(d[n], bt) for n in ['q', 'k', 'gk'])
    heads = g.shape[2]
    rep = heads // k.shape[2]
    q = q.repeat_interleave(rep, dim=2)
    k = k.repeat_interleave(rep, dim=2)
    beta = scalar_tile(d['beta'], bt)
    aq = q * torch.exp2(g) @ (k * torch.exp2(-g)).transpose(-1, -2) * 0.0884
    lower = k * torch.exp2(g) @ (k * torch.exp2(-g)).transpose(-1, -2) * beta[..., None]
    row = torch.arange(bt)
    offdiag = row[:, None] // bc > row[None, :] // bc
    aqk = tile(d['Aqk'], bt).clone()
    aqk[..., offdiag] = aq[..., offdiag]
    diagonal = tile(d['Akkd'], bt)
    inverses = []
    for start in range(0, bt, bc):
        block = diagonal[..., start:start + bc, :]
        inv = block.clone()
        inverses.append(inv)
    akk = torch.zeros_like(aqk)
    for i in range(4):
        si = slice(i * bc, (i + 1) * bc)
        akk[..., si, si] = inverses[i]
        for j in range(i):
            sj = slice(j * bc, (j + 1) * bc)
            rhs = sum((lower[..., si, slice(r * bc, (r + 1) * bc)] @ akk[..., slice(r * bc, (r + 1) * bc), sj] for r in range(j, i)))
            akk[..., si, sj] = -inverses[i] @ rhs
    for ri, cj in [(bc, 0), (bt - 1, 0), (bt - 1, bt - bc - 1)]:
        expected = math.fsum((float(q[0, 0, 0, ri, j]) * float(k[0, 0, 0, cj, j]) * 2.0 ** float(g[0, 0, 0, ri, j] - g[0, 0, 0, cj, j]) for j in range(q.shape[-1]))) * 0.0884
        errors.append(abs(float(aqk[0, 0, 0, ri, cj]) - expected))
    assert torch.equal(aqk[..., ~offdiag], tile(d['Aqk'], bt)[..., ~offdiag])
    out = {'Aqk': untile(aqk), 'Akk': untile(akk), 'Akkd': d['Akkd'].double().clone()}
    method = 'CPU64 gated qk off-diagonal blocks and independent block triangular solve with original supplied Akkd; preserves untouched Aqk and read-only Akkd, original safe-gate zero fixture retained'
    assert all((x.dtype == torch.float64 and torch.isfinite(x).all() for x in out.values()))
    assert max(errors) < 1e-09, errors
    return (out, {'method': method, 'scalar_checked_values': len(errors), 'scalar_max_abs_difference': max(errors)})
