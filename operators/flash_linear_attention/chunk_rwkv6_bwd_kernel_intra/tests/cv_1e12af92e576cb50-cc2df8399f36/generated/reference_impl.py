"""Existing independent CPU equations; file-generation main omitted."""
import torch
import reference_impl_builder as builder

def reference(cid, d):
    assert 'cv_1e12af92e576cb50' == 'cv_1e12af92e576cb50'
    bt = d['chunk_size']
    q, k, gi, ge, da = (builder.tile(d[n], bt) for n in ('q', 'k', 'gi', 'ge', 'dA'))
    da = torch.tril(da, diagonal=-1)
    dq = da @ (k * torch.exp2(-gi)) * torch.exp2(ge)
    dk = da.transpose(-1, -2) @ (q * torch.exp2(ge)) * torch.exp2(-gi)
    checks = []
    for b, chunk, head in ((0, 0, 0), (3, 3, 3)):
        qq = q[b, chunk, head].clone().requires_grad_()
        kk = k[b, chunk, head].clone().requires_grad_()
        weights = torch.exp2(ge[b, chunk, head, :, None, :] - gi[b, chunk, head, None, :, :])
        attention = (qq[:, None, :] * kk[None, :, :] * weights).sum(-1)
        grads = torch.autograd.grad((attention * da[b, chunk, head]).sum(), (qq, kk))
        for name, actual, expected in (('dq', dq[b, chunk, head], grads[0]), ('dk', dk[b, chunk, head], grads[1])):
            error = float((actual - expected).abs().max())
            assert error < 1e-09, (name, error)
            checks.append({'output': name, 'max_abs_error': error})
    return ({'dq': builder.untile(dq), 'dk': builder.untile(dk)}, {'method': 'CPU64 derivative of strictly causal sum(q_i*k_j*exp2(ge_i-gi_j))*dA_ij; independent autograd check', 'autograd_cross_checks': checks, 'original_inputs_unchanged': True})
