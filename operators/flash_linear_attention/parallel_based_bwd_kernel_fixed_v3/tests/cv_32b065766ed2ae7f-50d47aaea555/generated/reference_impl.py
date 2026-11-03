"""Existing independent CPU equations; file-generation main omitted."""
import torch

def errors(a, b):
    e = (a - b).abs()
    return dict(max_abs=float(e.max()), rms=float(e.square().mean().sqrt()))

def based_math(d):
    q, k, v, do, dz = [d[n].double() for n in ['q', 'k', 'v', 'do', 'dz']]
    s = d['scale']
    T = d['T']
    mask = torch.ones(T, T, dtype=torch.bool).tril()
    score = q @ k.transpose(-1, -2) * s
    p = (1 + score + 0.5 * score.square()).masked_fill(~mask, 0)
    deriv = ((do @ v.transpose(-1, -2) + dz[..., None]) * (1 + score)).masked_fill(~mask, 0)
    ref = dict(dq=(s * (deriv @ k)).unsqueeze(0), dk=(s * (deriv.transpose(-1, -2) @ q)).unsqueeze(0), dv=(p.transpose(-1, -2) @ do).unsqueeze(0))
    qa = q.clone().requires_grad_()
    ka = k.clone().requires_grad_()
    va = v.clone().requires_grad_()
    score2 = torch.einsum('bhtk,bhsk->bhts', qa, ka) * s
    p2 = (1 + score2 + 0.5 * score2.square()).masked_fill(~mask, 0)
    out = torch.einsum('bhts,bhsv->bhtv', p2, va)
    normalizer = p2.sum(-1)
    objective = (out * do).sum() + (normalizer * dz).sum()
    grads = torch.autograd.grad(objective, (qa, ka, va))
    checks = {name: errors(ref[name], grad.unsqueeze(0)) for name, grad in zip(('dq', 'dk', 'dv'), grads)}
    for name, grad in zip(('dq', 'dk', 'dv'), grads):
        assert torch.allclose(ref[name], grad.unsqueeze(0), atol=1e-10, rtol=1e-10)
    return (ref, checks)
