"""Existing independent CPU equations; file-generation main omitted."""
import math
import torch

def original_diag_reference(d):
    return diag_reference({'input_data': d})

def autograd_diag_check(d, outputs):
    bt = d['BT']
    checks = []
    for batch, chunk, head in [(0, 0, 0), (1, d['NT'] - 1, d['H'] - 1)]:
        start = chunk * bt
        sl = slice(start, start + bt)
        q = d['q'][batch, sl, 0].double().requires_grad_()
        k = d['k'][batch, sl, 0].double().requires_grad_()
        v = d['v'][batch, sl, head].double().requires_grad_()
        g = d['g'][batch, sl, head].double().requires_grad_()
        levels = d['l'][batch, sl, head].double().requires_grad_()
        weights = levels.gather(1, d['llut'].long())
        causal = torch.ones(bt, bt, dtype=torch.bool).tril()
        weights = torch.where(causal, weights, 0.0)
        a = q @ k.T * torch.exp(g[:, None] - g[None, :]) * weights
        loss = (a @ v * d['do'][batch, sl, head].double()).sum()
        grads = torch.autograd.grad(loss, (q, k, v, g, levels))
        for name, grad in zip(('dq', 'dk', 'dv', 'dg', 'dl'), grads):
            actual = outputs[name][batch, sl, head]
            if name != 'dl':
                actual = actual - d[name][batch, sl, head].double()
            else:
                written = int(math.log2(bt)) + 1
                assert torch.equal(actual[:, written:], d[name][batch, sl, head, written:].double())
                actual = actual[:, :written]
                grad = grad[:, :written]
            error = (actual - grad).abs().max().item()
            assert error < 1e-09, (name, error)
            checks.append({'output': name, 'max_abs_difference': error})
    return checks

def reference(cid, d):
    assert d['G'] == 1 and d['T'] % d['BT'] == 0
    out = original_diag_reference(d)
    checks = autograd_diag_check(d, out)
    return out

from original_cpu import diag_reference
