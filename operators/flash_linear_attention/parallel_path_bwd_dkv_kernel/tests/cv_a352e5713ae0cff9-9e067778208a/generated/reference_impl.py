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

def dkv_reference(d):
    assert d['g_cumsum'] is None and d['dg_cumsum'] is None
    q, k, v, do, ell, delta = (d[key].double() for key in ('q_new_large', 'k_new_large', 'v', 'do', 'L', 'delta'))
    b, t, blocks, hq, width = q.shape
    h, bt, bs, split = (k.shape[2], d['BT'], d['BS'], d['S'])
    assert width == v.shape[-1] and split % bt == 0 and (split % bs == 0)
    dk, dv = (torch.zeros(b, t, hq, width, dtype=torch.float64), d['dv'].double().clone())
    checks, pullbacks = ([], [])
    for bi in range(b):
        for hi in range(hq):
            kh = hi // (hq // h)
            for start in range(0, t, bt):
                end = min(start + bt, t)
                key, value = (k[bi, start:end, kh], v[bi, start:end, kh])
                group = start // split + 1
                for offset in range(((t + bs - 1) // bs - 1) * bs, (start // split + 1) * split - bs, -bs):
                    query, grad = (q[bi, offset:offset + bs, group, hi], do[bi, offset:offset + bs, hi])
                    score = mm(key, query.T, checks)
                    prob = torch.exp2(score * (d['scale'] * 1.44269504) - ell[bi, offset:offset + bs, hi][None, :])
                    dp = mm(value, grad.T, checks)
                    da = (dp - delta[bi, offset:offset + bs, hi][None, :]) * prob * d['scale']
                    grad_k, grad_v = (mm(da, query, checks), mm(prob, grad, checks))
                    dk[bi, start:end, hi] += grad_k
                    dv[bi, start:end, hi] += grad_v
                    if len(pullbacks) < 4:
                        ka, va = (key.clone().requires_grad_(), value.clone().requires_grad_())
                        grads = torch.autograd.grad((ka @ query.T * da).sum() + (va @ grad.T * prob).sum(), (ka, va))
                        for expected, actual in zip((grad_k, grad_v), grads):
                            torch.testing.assert_close(expected, actual, rtol=1e-10, atol=1e-10)
                        pullbacks.append(max(float((grad_k - grads[0]).abs().max()), float((grad_v - grads[1]).abs().max())))
    return (dict(dk=dk, dv=dv), dict(method='CPU64 inter-split attention gradient; original pre-existing dv added and final key split writes zero dk', scalar_dot_errors=checks, local_autograd_pullback_errors=pullbacks, preserved_input_outputs=['dv'], none_outputs=['dg_cumsum']))
