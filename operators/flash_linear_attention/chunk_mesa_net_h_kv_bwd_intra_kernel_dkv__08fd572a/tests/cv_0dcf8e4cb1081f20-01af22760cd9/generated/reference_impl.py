"""Existing independent CPU equations; file-generation main omitted."""
import torch

def mesa_reference(data, source=None):
    common = [data[k].double() for k in ('q_star', 'k', 'v', 'beta', 'h_kv', 'g', 'do')]
    dims = [data[k] for k in ('B', 'T', 'H', 'K', 'V', 'BT')]
    dk, dv, dg = naive_chunk_mesa_net_h_kv_bwd_intra_dkv(*common, data['dh_kv'].double(), *dims)
    dq, final = naive_chunk_mesa_net_h_kv_bwd_intra_dq(*common, dg, *dims)
    outputs = dict(dk=dk, dv=dv, dg=dg, dq=dq, dg_final=final)
    bt = data['BT']
    q = data['q_star'][0, :bt, 0].double().requires_grad_()
    kb = (data['k'][0, :bt, 0].double() * data['beta'][0, :bt, 0].double()[:, None]).requires_grad_()
    v = data['v'][0, :bt, 0].double().requires_grad_()
    g = data['g'][0, :bt, 0].double().requires_grad_()
    h, dh, do = (data['h_kv'][0, 0].double(), data['dh_kv'][0, 0].double(), data['do'][0, :bt, 0].double())
    decay = torch.tril(torch.exp(g[:, None] - g[None, :]))
    inter = torch.exp(g)[:, None] * (q @ h)
    output = inter + q @ kb.T * decay @ v
    state = torch.exp(g[-1]) * h + (kb * torch.exp(g[-1] - g)[:, None]).T @ v
    grads = torch.autograd.grad((output * do).sum() + (state * dh).sum(), (q, kb, v, g))
    independent = dict(dq=grads[0], dk=grads[1], dv=grads[2], dg_final=grads[3], dg=grads[3] - (inter.detach() * do).sum(-1))
    differences = {}
    for key, gradient in independent.items():
        expected = outputs[key][0, :bt, 0]
        torch.testing.assert_close(expected, gradient, rtol=1e-10, atol=1e-10)
        differences[key] = float((expected - gradient).abs().max())
    return outputs

def reference(case, contract, data):
    cid = case['id']
    return mesa_reference(data)

from original_cpu import naive_chunk_mesa_net_h_kv_bwd_intra_dkv
from original_cpu import naive_chunk_mesa_net_h_kv_bwd_intra_dq
