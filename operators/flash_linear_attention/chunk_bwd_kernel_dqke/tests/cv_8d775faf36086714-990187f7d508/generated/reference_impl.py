"""Existing independent CPU equations; file-generation main omitted."""
import torch

def reference(inputs):
    values = naive_chunk_bwd_kernel_dqke(*(inputs[n] for n in ('q', 'k', 'v_new', 'eta', 'h', 'do_tensor', 'dh', 'dhb')), float(inputs['scale']), int(inputs['chunk_size']))
    refs = dict(zip(('dq', 'dk', 'de'), values))
    assert all((v.dtype == torch.float64 and torch.isfinite(v).all() for v in refs.values()))
    sl = slice(0, int(inputs['chunk_size']))
    q, k, e = (inputs[n][0, sl, 0].double().clone().requires_grad_() for n in ('q', 'k', 'eta'))
    v = inputs['v_new'][0, sl, 0].double()
    h = inputs['h'][0, 0, 0].double()
    do = inputs['do_tensor'][0, sl, 0].double()
    dh = inputs['dh'][0, 0, 0].double()
    dhb = inputs['dhb'][0, 0, 0, 0].double()
    scale = float(inputs['scale'])
    out = q @ h * scale - e * (torch.tril(q @ k.T * scale + torch.ones(q.shape[0], q.shape[0], dtype=torch.float64)) @ v)
    final_h = h - e[-1] * (k.T @ v)
    final_b = -e[-1] * v.sum(0)
    grads = torch.autograd.grad((out * do).sum() + (final_h * dh).sum() + (final_b * dhb).sum(), (q, k, e))
    errors = {}
    for name, gold in zip(refs, grads):
        expected = refs[name][0, sl, 0]
        torch.testing.assert_close(expected, gold, rtol=1e-10, atol=1e-10)
        errors[name] = float((expected - gold).abs().max())
    return refs

from original_cpu import naive_chunk_bwd_kernel_dqke
