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

def intra_reference(d):
    assert d['g_cumsum'] is None and d['dg_cumsum'] is None
    q, k, v, w1, w2, do, ell, delta = (d[key].double() for key in ('q_new', 'k_new', 'v', 'w', 'h', 'do', 'L', 'delta'))
    b, t, hq, width = q.shape
    h, bt, split = (k.shape[2], d['BT'], d['S'])
    dq = torch.empty_like(q)
    outputs = {key: d[key].double().clone() for key in ('dk', 'dv', 'dw1', 'dw2')}
    checks, pullbacks = ([], [])
    for bi in range(b):
        for hi in range(hq):
            kh = hi // (hq // h)
            for start in range(0, t, bt):
                end = min(start + bt, t)
                grad_q = d['dq'][bi, start:end, hi].double().clone()
                query, grad = (q[bi, start:end, hi], do[bi, start:end, hi])
                for offset in range(start // split * split, start, bt):
                    key, value = (k[bi, offset:offset + bt, kh], v[bi, offset:offset + bt, kh])
                    transformed = query.clone()
                    for middle in range(start - bt, offset, -bt):
                        a, c = (w1[bi, middle:middle + bt, kh], w2[bi, middle:middle + bt, kh])
                        transformed -= mm(mm(transformed, a.T, checks), c, checks)
                    score = mm(transformed, key.T, checks)
                    prob = torch.exp2(score * (d['scale'] * 1.44269504) - ell[bi, start:end, hi][:, None])
                    da = (mm(grad, value.T, checks) - delta[bi, start:end, hi][:, None]) * prob * d['scale']
                    outputs['dv'][bi, offset:offset + bt, hi] += mm(prob.T, grad, checks)
                    outputs['dk'][bi, offset:offset + bt, hi] += mm(da.T, transformed, checks)
                    a, c = (w1[bi, offset:offset + bt, kh], w2[bi, offset:offset + bt, kh])
                    grad_a = mm(grad_q, c.T, checks)
                    activation = mm(transformed, a.T, checks)
                    grad_c = -mm(activation.T, grad_q, checks)
                    grad_w = -mm(grad_a.T, transformed, checks)
                    next_grad_q = grad_q - mm(grad_a, a, checks) + mm(da, key, checks)
                    if len(pullbacks) < 4:
                        qa, aa, ca = (transformed.clone().requires_grad_(), a.clone().requires_grad_(), c.clone().requires_grad_())
                        objective = ((qa - qa @ aa.T @ ca) * grad_q).sum() + (qa @ key.T * da).sum()
                        grads = torch.autograd.grad(objective, (qa, aa, ca))
                        for expected, actual in zip((next_grad_q, grad_w, grad_c), grads):
                            torch.testing.assert_close(expected, actual, rtol=1e-10, atol=1e-10)
                        pullbacks.append(max((float((e - a).abs().max()) for e, a in zip((next_grad_q, grad_w, grad_c), grads))))
                    outputs['dw2'][bi, offset:offset + bt, hi] += grad_c
                    outputs['dw1'][bi, offset:offset + bt, hi] += grad_w
                    grad_q = next_grad_q
                dq[bi, start:end, hi] = grad_q
    return (dict(dq_new=dq, **outputs), dict(method='CPU64 intra-split attention and transform pullback; original dq/dk/dv/dw1/dw2 pre-state preserved', scalar_dot_errors=checks, local_autograd_pullback_errors=pullbacks, preserved_input_outputs=['dq', 'dk', 'dv', 'dw1', 'dw2'], none_outputs=['dg_cumsum']))
