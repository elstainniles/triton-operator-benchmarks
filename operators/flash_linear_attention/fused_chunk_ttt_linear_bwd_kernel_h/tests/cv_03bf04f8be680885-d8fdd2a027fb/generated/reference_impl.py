"""Existing independent CPU equations; file-generation main omitted."""
import torch

def original_naive(folder, name):
    return (naive_chunk_ttt_linear_bwd_h, {'function': 'naive_chunk_ttt_linear_bwd_h', 'function_AST_unmodified': True})

def reference(cid, d):
    fn, origin = original_naive('chunk_ttt_linear_bwd_kernel_h', 'naive_chunk_ttt_linear_bwd_h')
    eps, bt = (d['eps'], d['BT'])
    values = fn(*(d[n] for n in ('k', 'v', 'w', 'b', 'eta')), eps, d['h0'], d['hb0'], bt)
    out = dict(zip(('h', 'v2', 'x', 'y', 'r'), values))
    B, T, H, D = d['k'].shape
    N = T // bt

    def tiles(x):
        return x.double().reshape(B, N, bt, H, D).permute(0, 1, 3, 2, 4)
    do, k, v2 = (tiles(d['do']), tiles(d['k']), tiles(out['v2']))
    eta = d['eta'].double().reshape(B, N, bt, H, 1).permute(0, 1, 3, 2, 4)
    dq = (do @ out['h'].transpose(-1, -2) - torch.tril(do @ v2.transpose(-1, -2)) @ k * eta) * d['scale']
    out['dq'] = dq.permute(0, 1, 3, 2, 4).reshape(B, T, H, D)
    q = d['k'][0, :bt, 0].double().clone().requires_grad_()
    kk = k[0, 0, 0]
    vv = v2[0, 0, 0]
    hh = out['h'][0, 0, 0]
    oo = (q @ hh - torch.tril(q @ kk.T) @ vv * eta[0, 0, 0]) * d['scale']
    grad = torch.autograd.grad((oo * do[0, 0, 0]).sum(), q)[0]
    torch.testing.assert_close(grad, dq[0, 0, 0], rtol=1e-10, atol=1e-10)
    origin['dq_forward_autograd_max_error'] = float((grad - dq[0, 0, 0]).abs().max())
    return (out, {'method': 'Original unmodified naive CPU64 state/intermediate reference; independent q-adjoint for fused bwd_h', **origin})

from original_cpu import naive_chunk_ttt_linear_bwd_h
