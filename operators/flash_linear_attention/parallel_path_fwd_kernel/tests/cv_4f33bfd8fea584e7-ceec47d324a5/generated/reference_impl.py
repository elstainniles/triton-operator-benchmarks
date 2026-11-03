"""Existing independent CPU equations; file-generation main omitted."""
import torch
import math

def mm(a, b, checks):
    value = a @ b
    if len(checks) < 12 and value.numel():
        for i, j in ((0, 0), (a.shape[0] - 1, b.shape[1] - 1)):
            scalar = math.fsum((float(a[i, k]) * float(b[k, j]) for k in range(a.shape[1])))
            error = abs(float(value[i, j]) - scalar)
            assert error <= 1e-09 * (1 + abs(scalar)), (error, scalar)
            checks.append(error)
    return value

def fwd_reference(d):
    assert d['cu_seqlens'] is None and d['g_cumsum'] is None and (d['BT'] == d['BS'])
    q, k, v, w1, w2, o, ell, m = (d[key].double() for key in ('q', 'k', 'v', 'w1', 'w2', 'o', 'L', 'M'))
    b, t, hq, _ = q.shape
    h, bt, bs = (k.shape[2], d['BT'], d['BS'])
    outputs, normalizers = (torch.empty_like(o), torch.empty_like(ell))
    checks, reductions = ([], [])
    for bi in range(b):
        for hi in range(hq):
            kh = hi // (hq // h)
            for start in range(0, t, bt):
                query = q[bi, start:start + bt, hi].clone()
                result = o[bi, start:start + bt, hi].clone()
                lse = ell[bi, start:start + bt, hi].clone()
                maximum = m[bi, start:start + bt, hi].clone()
                logits, values = ([], [])
                for offset in range(start - bs, -bs, -bs):
                    key, value = (k[bi, offset:offset + bs, kh], v[bi, offset:offset + bs, kh])
                    score = mm(query, key.T, checks) * (d['scale'] * 1.44269504)
                    logits.append(score)
                    values.append(value)
                    newmax = torch.maximum(maximum, score.max(-1).values)
                    alpha = torch.exp2(maximum - newmax)
                    probabilities = torch.exp2(score - newmax[:, None])
                    result = result * alpha[:, None] + mm(probabilities, value, checks)
                    lse = lse * alpha + probabilities.sum(-1)
                    maximum = newmax
                    weight1, weight2 = (w1[bi, offset:offset + bs, kh], w2[bi, offset:offset + bs, kh])
                    query = query - mm(mm(query, weight1.T, checks), weight2, checks)
                outputs[bi, start:start + bt, hi] = result / lse[:, None]
                normalizers[bi, start:start + bt, hi] = torch.log2(lse) + maximum
                if start in (0, t - bt):
                    initial_m = m[bi, start:start + bt, hi]
                    initial_l = ell[bi, start:start + bt, hi]
                    initial_o = o[bi, start:start + bt, hi]
                    if logits:
                        scores = torch.cat(logits, -1)
                        common = torch.maximum(initial_m, scores.max(-1).values)
                        p = torch.exp2(scores - common[:, None])
                        numerator = initial_o * torch.exp2(initial_m - common)[:, None] + p @ torch.cat(values, 0)
                        denominator = initial_l * torch.exp2(initial_m - common) + p.sum(-1)
                    else:
                        common, numerator, denominator = (initial_m, initial_o, initial_l)
                    direct = numerator / denominator[:, None]
                    direct_l = torch.log2(denominator) + common
                    torch.testing.assert_close(direct, outputs[bi, start:start + bt, hi], rtol=1e-10, atol=1e-10)
                    torch.testing.assert_close(direct_l, normalizers[bi, start:start + bt, hi], rtol=1e-10, atol=1e-10)
                    reductions.append(float((direct - outputs[bi, start:start + bt, hi]).abs().max()))
    return (dict(o_new=outputs, L_new=normalizers), dict(method='CPU64 sequential query transforms and attention using original o/L/M state; online normalization crosschecked by one-shot reduction', scalar_dot_errors=checks, online_vs_direct_errors=reductions))
