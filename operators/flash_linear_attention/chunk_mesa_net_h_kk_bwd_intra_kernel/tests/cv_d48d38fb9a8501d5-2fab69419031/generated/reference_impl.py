"""Existing independent CPU equations; file-generation main omitted."""
import torch

def load_reference(source, name, promote=False, transpose_ttt=False):
    return (naive_chunk_mesa_net_h_kk_bwd_intra, [])

def mesa(d, source):
    fn, _ = load_reference(source, 'naive_chunk_mesa_net_h_kk_bwd_intra', promote=True)
    values = fn(*(d[key].double() for key in ('k', 'beta', 'g', 'h', 'dh', 'q_star', 'dq', 'dk_beta')), chunk_size=d['BT'])
    out = dict(zip(('dk', 'dg', 'dbeta', 'dlamb'), values))
    bt = d['BT']
    k = d['k'][0, :bt, 0].double().requires_grad_()
    beta = d['beta'][0, :bt, 0].double().requires_grad_()
    g = d['g'][0, :bt, 0].double().requires_grad_()
    q, dq = (d['q_star'][0, :bt, 0].double(), d['dq'][0, :bt, 0].double())
    h, dh = (d['h'][0, 0].double(), d['dh'][0, 0].double())
    dk_beta = d['dk_beta'][0, :bt, 0].double()
    kb = k * beta[:, None]
    decay = torch.tril(torch.exp(g[:, None] - g[None, :]))
    output = torch.exp(g)[:, None] * (q @ h.T) + q @ kb.T * decay @ k
    state = torch.exp(g[-1]) * h + (kb * torch.exp(g[-1] - g)[:, None]).T @ k
    objective = (output * dq).sum() + (state * dh).sum() - (kb * dk_beta).sum()
    gradients = torch.autograd.grad(objective, (k, beta, g))
    errors = {}
    for key, gradient in zip(('dk', 'dbeta', 'dg'), gradients):
        actual = out[key][0, :bt, 0]
        torch.testing.assert_close(actual, -gradient, rtol=1e-10, atol=1e-10)
        errors[key] = float((actual + gradient).abs().max())
    expected_lambda = -(q * dq).sum(0)
    torch.testing.assert_close(out['dlamb'][0, 0, 0], expected_lambda, rtol=0, atol=0)
    return (out, dict(method='Original independent Mesa hkk naive equations in FP64; autograd of attention/state loss confirms dk/dbeta/dg, explicit lambda reduction', reference_dtype_changes='float32 arithmetic/default zeros promoted to float64', autograd_first_chunk_max_abs=errors))

from original_cpu import naive_chunk_mesa_net_h_kk_bwd_intra
