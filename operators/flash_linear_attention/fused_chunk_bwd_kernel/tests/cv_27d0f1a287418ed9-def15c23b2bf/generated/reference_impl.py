"""Existing independent CPU equations; file-generation main omitted."""
from reference_impl_tile import tile
from reference_impl_scalar_tile import scalar_tile
from reference_impl_untile import untile
import torch

def reference(cid, d):
    assert d['g_gamma'] is None and d['g'] is not None
    bt = 32
    q, k, v = (tile(d[n], bt).requires_grad_() for n in ['q', 'k', 'v'])
    g = scalar_tile(d['g'], bt).requires_grad_()
    h0 = d['initial_state'].double().requires_grad_()
    state = h0
    do = tile(d['do'], bt)
    loss = torch.zeros((), dtype=torch.float64)
    for index in range(q.shape[1]):
        qi, ki, vi, gi = (q[:, index], k[:, index], v[:, index], g[:, index])
        causal = torch.tril(qi @ ki.transpose(-1, -2) * torch.exp(gi[..., None] - gi[..., None, :]))
        oi = (torch.exp(gi)[..., None] * (qi @ state) + causal @ vi) * d['scale']
        loss = loss + (oi * do[:, index]).sum()
        state = torch.exp(gi[..., -1, None, None]) * state + (ki * torch.exp(gi[..., -1, None] - gi)[..., None]).transpose(-1, -2) @ vi
    loss = loss + (state * d['dht'].double()).sum()
    dq, dk, dv, dg_cumulative, dh0 = torch.autograd.grad(loss, (q, k, v, g, h0))
    dg = dg_cumulative.flip(-1).cumsum(-1).flip(-1)
    outputs = {'dq': untile(dq), 'dk': untile(dk), 'dv': untile(dv), 'dg': untile(dg[..., None]).squeeze(-1), 'dh0': dh0}
    expected = d['dht'].double().clone()
    raw = d['g'].double().reshape(d['g'].shape[0], -1, bt, d['g'].shape[-1]).clone()
    raw[:, :, 1:] -= raw[:, :, :-1].clone()
    raw = raw.reshape_as(d['g'])
    for t in range(d['q'].shape[1] - 1, -1, -1):
        expected = (expected + d['q'][:, t].double()[..., None] * d['do'][:, t].double()[..., None, :] * d['scale']) * torch.exp(raw[:, t])[..., None, None]
    error = float((expected - dh0).abs().max())
    assert error < 1e-08, error
    assert all((x.dtype == torch.float64 and torch.isfinite(x).all() for x in outputs.values()))
    return (outputs, {'method': 'CPU FP64 automatic differentiation of gated linear-attention forward recurrence and final-state loss; convert cumulative-g derivative to token-g derivative; independent token adjoint check of dh0', 'token_adjoint_dh0_max_abs_error': error})
