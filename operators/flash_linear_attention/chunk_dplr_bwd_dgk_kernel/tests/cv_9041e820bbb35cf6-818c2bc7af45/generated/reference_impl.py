"""Existing independent CPU equations; file-generation main omitted."""
from reference_impl_tile import tile
import math
from reference_impl_untile import untile
import torch

def reference(cid, d):
    bt = d['chunk_size']
    checks = []
    assert d['cu_seqlens'] is None and d['chunk_indices'] is None and (d['safe_gate'] is True)
    q, k, a, b, gi, ge, dqg, dkg, dag, dbg = (tile(d[n], bt) for n in ('q', 'k', 'a', 'b', 'gi', 'ge', 'dqg', 'dkg', 'dag', 'dbg'))
    aqk, aqb, aak, aab = (tile(d[n], bt).tril(diagonal=0 if n in ('dAqk', 'dAqb') else -1) for n in ('dAqk', 'dAqb', 'dAak', 'dAab'))
    ei, ii, ee = (torch.exp2(gi), torch.exp2(-gi), torch.exp2(ge))
    qt, kt, at, bb = (q * ei, k * ii, a * ee, b * ii)
    dq = (aqk @ kt + aqb @ bb) * ei + dqg * ei * d['scale']
    da = (aak @ kt + aab @ bb) * ee + dag * ee
    decay = torch.exp2(gi[..., -1:, :] - gi)
    dk = (aqk.transpose(-1, -2) @ qt + aak.transpose(-1, -2) @ at) * ii + dkg * decay
    db = (aqb.transpose(-1, -2) @ qt + aab.transpose(-1, -2) @ at) * ii + dbg * decay
    intermediate = dq * q + da * a - dk * k - db * b
    offset = da * a
    last = d['dgk_last'].double().reshape(*q.shape[:3], q.shape[-1])
    dg = intermediate.flip(-2).cumsum(-2).flip(-2) + last[..., None, :] - offset
    qa, ka, aa, ba = (x.detach().requires_grad_() for x in (q, k, a, b))
    qexp, kexp, aexp, bexp = (qa * ei, ka * ii, aa * ee, ba * ii)
    objective = (qexp @ kexp.transpose(-1, -2) * aqk).sum() + (qexp @ bexp.transpose(-1, -2) * aqb).sum()
    objective += (aexp @ kexp.transpose(-1, -2) * aak).sum() + (aexp @ bexp.transpose(-1, -2) * aab).sum()
    objective += (qexp * dqg * d['scale']).sum() + (aexp * dag).sum() + (ka * decay * dkg).sum() + (ba * decay * dbg).sum()
    auto = torch.autograd.grad(objective, (qa, ka, aa, ba))
    checks.extend((float((value - gold).abs().max()) for value, gold in zip((dq, dk, da, db), auto)))
    for row, col in ((0, 0), (bt // 2, 3), (bt - 1, q.shape[-1] - 1)):
        scalar = math.fsum((float(dq[0, 0, 0, t, col]) * float(q[0, 0, 0, t, col]) + float(da[0, 0, 0, t, col]) * float(a[0, 0, 0, t, col]) - float(dk[0, 0, 0, t, col]) * float(k[0, 0, 0, t, col]) - float(db[0, 0, 0, t, col]) * float(b[0, 0, 0, t, col]) for t in range(row, bt)))
        scalar += float(last[0, 0, 0, col]) - float(da[0, 0, 0, row, col]) * float(a[0, 0, 0, row, col])
        checks.append(abs(float(dg[0, 0, 0, row, col]) - scalar))
    out = {'dq': untile(dq), 'dk': untile(dk), 'da': untile(da), 'db': untile(db), 'dgk_output': untile(dg)}
    method = 'Independent CPU64 causal inclusive/strict bilinear gradients plus reverse prefix sum for dgk; autograd verifies all four matrix derivatives and scalar suffix sums verify actual target output'
    assert all((x.dtype == torch.float64 and torch.isfinite(x).all() for x in out.values()))
    assert checks and max(checks) < 1e-08, checks
    return (out, {'method': method, 'scalar_checked_values': len(checks), 'scalar_max_abs_difference': max(checks), 'all_defined_original_outputs_compared': list(out), 'reference_unrounded': True})
