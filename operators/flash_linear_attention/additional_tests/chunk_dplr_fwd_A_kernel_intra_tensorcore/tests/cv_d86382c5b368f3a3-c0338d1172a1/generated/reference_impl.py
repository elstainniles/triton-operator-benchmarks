"""Existing independent CPU equations; file-generation main omitted."""
import torch
import math

def dplr(d):
    q, k, a, b, gi, ge = (d[key].double() for key in ('q', 'k', 'a', 'b', 'gi', 'ge'))
    batch, length, heads, width = q.shape
    bt = d['chunk_size']
    matrices = {key: torch.zeros(batch, length, heads, bt, dtype=torch.float64) for key in ('Aab', 'Aqk', 'Aak', 'Aqb')}
    vectors = {key: torch.empty_like(q) for key in ('qg', 'kg', 'ag', 'bg')}
    errors = []
    for bi in range(batch):
        for hi in range(heads):
            for start in range(0, length, bt):
                end = min(start + bt, length)
                qq, kk, aa, bb = (x[bi, start:end, hi] for x in (q, k, a, b))
                g, e = (gi[bi, start:end, hi], ge[bi, start:end, hi])
                qg, ag = (qq * d['scale'] * torch.exp2(g), aa * torch.exp2(e))
                ki, b_inv = (kk * torch.exp2(-g), bb * torch.exp2(-g))
                values = dict(Aqk=torch.tril(qg @ ki.T), Aqb=torch.tril(qg @ b_inv.T), Aak=torch.tril(ag @ ki.T, diagonal=-1), Aab=torch.tril(ag @ b_inv.T, diagonal=-1))
                for key, value in values.items():
                    matrices[key][bi, start:end, hi, :end - start] = value
                vectors['qg'][bi, start:end, hi] = qg
                vectors['ag'][bi, start:end, hi] = ag
                vectors['kg'][bi, start:end, hi] = kk * torch.exp2(g[-1] - g)
                vectors['bg'][bi, start:end, hi] = bb * torch.exp2(g[-1] - g)
                if len(errors) < 24:
                    row, col = (min(1, end - start - 1), 0)
                    for key, left, right, gain, factor in [('Aqk', qq, kk, g, d['scale']), ('Aqb', qq, bb, g, d['scale']), ('Aak', aa, kk, e, 1), ('Aab', aa, bb, e, 1)]:
                        scalar = factor * math.fsum((float(left[row, dim]) * float(right[col, dim]) * 2 ** float(gain[row, dim] - g[col, dim]) for dim in range(width)))
                        actual = float(values[key][row, col])
                        error = abs(actual - scalar)
                        assert error <= 1e-10 * (1 + abs(scalar))
                        errors.append(error)
    return ({**matrices, **vectors}, dict(method='CPU64 gated bilinear forms with inclusive qk/qb and strict ak/ab masks; uncentered equations cancel the temporary centering offset analytically', scalar_gate_dot_checks=len(errors), scalar_max_abs_difference=max(errors)))
