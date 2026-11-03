"""Existing independent CPU equations; file-generation main omitted."""
import torch

def reference(cid, d):
    bt, s = (d['BT'], d['S'])
    b, t, h, k = d['k'].shape
    assert t % s == s % bt == 0
    ns = t // s
    nc = s // bt

    def blocks(value, heads=h):
        return value.double().reshape(b, ns, nc, bt, heads, k).permute(0, 1, 4, 2, 3, 5)

    def unblocks(value):
        return value.permute(0, 1, 3, 4, 2, 5).reshape(b, t, value.shape[2], k)
    w1, w2, key = (blocks(d[n]) for n in ['w1', 'w2', 'k'])
    identity = torch.eye(k, dtype=torch.float64)
    errors = []
    hq = d['dk'].shape[2]
    group = hq // h
    w1, w2, key = (x.repeat_interleave(group, dim=2) for x in (w1, w2, key))
    dk = blocks(d['dk'], hq)
    suffix = d['hc_suffix'].double().reshape(b, ns, nc, h, k, k).permute(0, 1, 3, 2, 4, 5).repeat_interleave(group, dim=2)
    carry = d['dhc_whole'].double().reshape(b, ns, hq, k, k)
    dw1, dw2, dknew = (torch.empty_like(key) for _ in range(3))
    for j in range(nc):
        first = w1[:, :, :, j].detach().requires_grad_()
        second = w2[:, :, :, j].detach().requires_grad_()
        ki = key[:, :, :, j].detach().requires_grad_()
        hi = suffix[:, :, :, j].detach().requires_grad_()
        transformed_key = ki @ (identity - hi).transpose(-1, -2)
        state_update = hi + (identity - hi) @ first.transpose(-1, -2) @ second
        loss = (state_update * carry).sum() + (transformed_key * dk[:, :, :, j]).sum()
        grad1, grad2, gradk, gradstate = torch.autograd.grad(loss, (first, second, ki, hi))
        dw1[:, :, :, j] = grad1
        dw2[:, :, :, j] = grad2
        dknew[:, :, :, j] = gradk
        expected = (identity - hi.detach().transpose(-1, -2)) @ carry
        errors.extend([float((grad1 - second.detach() @ expected.transpose(-1, -2)).abs().max()), float((grad2 - first.detach() @ expected).abs().max())])
        carry = gradstate
    out = {'dw1': unblocks(dw1), 'dw2': unblocks(dw2), 'dk_new': unblocks(dknew)}
    method = 'Independent CPU64 autograd of each Householder state update plus transformed-key objective, chained state adjoint with exact supplied hc_suffix and GQA expansion'
    assert all((v.dtype == torch.float64 and torch.isfinite(v).all() for v in out.values()))
    assert max(errors) < 1e-08, errors
    return (out, {'method': method, 'independent_cross_checks': len(errors), 'max_cross_check_error': max(errors)})
