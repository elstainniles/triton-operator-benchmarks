"""Existing independent CPU equations; file-generation main omitted."""
import torch
import reference_impl_builder as builder

def reference(cid, d):
    bt = d['BT']
    assert d['G'] == 1 and d['T'] % bt == 0 and (d['cu_seqlens'] is None)
    B, NT, H, K, V = (d['B'], d['NT'], d['H'], d['K'], d['V'])
    v = builder.tile(d['v'], bt)
    k = builder.tile(d['k'], bt).expand(B, NT, H, bt, K)
    g = builder.scalar_tile(d['g'], bt)
    dh = d['dh'].double()
    dk = torch.exp(g[..., -1:] - g)[..., None] * (v @ dh.transpose(-1, -2))
    dot = (k * dk).sum(-1)
    dg = builder.scalar_tile(d['dg'], bt) - dot
    dg[..., -1] += dot.sum(-1) + d['dg_last'].double() * g[..., -1].exp()
    out = {'dk': builder.untile(dk), 'dg': builder.untile(dg[..., None]).squeeze(-1), 'dg_last': d['dg_last'].double().clone()}
    checks = []
    for batch, chunk, head in [(0, 0, 0), (1, NT - 1, H - 1)]:
        kk = k[batch, chunk, head].clone().requires_grad_()
        gg = g[batch, chunk, head].clone().requires_grad_()
        vv = v[batch, chunk, head]
        state = (kk * torch.exp(gg[-1] - gg)[:, None]).T @ vv
        loss = (state * dh[batch, chunk, head]).sum() + torch.exp(gg[-1]) * d['dg_last'][batch, chunk, head].double()
        gradk, gradg = torch.autograd.grad(loss, (kk, gg))
        for name, actual, expected in [('dk', dk[batch, chunk, head], gradk), ('dg', dg[batch, chunk, head] - builder.scalar_tile(d['dg'], bt)[batch, chunk, head], gradg)]:
            error = (actual - expected).abs().max().item()
            assert error < 1e-09, (name, error)
            checks.append({'output': name, 'max_abs_difference': error})
    return out

from original_cpu import diag_reference
