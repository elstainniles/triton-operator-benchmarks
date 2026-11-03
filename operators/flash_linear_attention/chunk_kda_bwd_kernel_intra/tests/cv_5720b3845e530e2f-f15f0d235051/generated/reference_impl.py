"""Existing independent CPU equations; file-generation main omitted."""
import math
import reference_impl_builder as builder
import torch

def intra_reference(d):
    b, t, heads, width = d['k'].shape
    hv = d['g'].shape[2]
    assert heads == hv and (not d['safe_gate'])
    bt, bk = (d['chunk_size'], 32)
    out = {name: torch.empty_like(d[source], dtype=torch.float64) for name, source in [('dq2', 'dq'), ('dk2', 'dk'), ('dg2', 'dg')]}
    out['db2'] = torch.empty((math.ceil(width / bk), *d['beta'].shape), dtype=torch.float64)
    errors = []
    for start in range(0, t, bt):
        stop = min(t, start + bt)
        n = stop - start

        def part(name):
            return d[name][:, start:stop].double().transpose(1, 2)
        q, k, g = (part(name) for name in ('q', 'k', 'g'))
        beta = part('beta')
        aq, ak = (torch.tril(part(name)[..., :n]) for name in ('dAqk', 'dAkk'))
        eg, emg = (torch.exp2(g), torch.exp2(-g))
        qgrad = aq @ (k * emg) * eg
        kleft = ak @ (k * emg) * eg
        kright = (aq.transpose(-1, -2) @ (q * eg) + ak.transpose(-1, -2) @ (k * beta[..., None] * eg)) * emg
        out['dq2'][:, start:stop] = (qgrad + part('dq')).transpose(1, 2)
        out['dk2'][:, start:stop] = (kleft * beta[..., None] + kright + part('dk')).transpose(1, 2)
        out['dg2'][:, start:stop] = (q * qgrad + (kleft * beta[..., None] - kright) * k + part('dg')).transpose(1, 2)
        for ki in range(math.ceil(width / bk)):
            sl = slice(ki * bk, min(width, (ki + 1) * bk))
            out['db2'][ki, :, start:stop] = (kleft[..., sl] * k[..., sl]).sum(-1).transpose(1, 2)
        if start in (0, t // bt * bt):
            qq, kk, gg, bb = [x[0, 0].clone().requires_grad_() for x in (q, k, g, beta)]
            weights = torch.exp2(gg[:, None] - gg[None, :])
            loss = ((qq[:, None] * kk[None, :] * weights).sum(-1) * aq[0, 0]).sum()
            loss += ((kk[:, None] * kk[None, :] * weights).sum(-1) * bb[:, None] * ak[0, 0]).sum()
            dq, dk, dg, db = torch.autograd.grad(loss, (qq, kk, gg, bb))
            actual = (qgrad[0, 0], (kleft * beta[..., None] + kright)[0, 0], (q * qgrad + (kleft * beta[..., None] - kright) * k)[0, 0], (kleft * k).sum(-1)[0, 0])
            expected = (dq, dk, dg / math.log(2), db)
            errors.extend((float((a - e).abs().max()) for a, e in zip(actual, expected)))
    assert max(errors) < 1e-08, errors
    return (out, {'method': 'CPU64 derivatives of causal qk and beta-weighted kk Gram matrices; preserve dq/dk/dg inputs and BK=32 db partials, including T=500 tail', 'autograd_max_abs_error': max(errors)})

def fused_reference(d):
    bt = d['chunk_size']
    q, k, v, vn, g, a, do, dv = (builder.tile(d[n], bt) for n in ('q', 'k', 'v', 'v_new', 'g', 'A', 'do', 'dv'))
    beta = builder.scalar_tile(d['beta'], bt)
    b, count, hv, _, width = g.shape
    assert q.shape[2] == hv and (not d['state_v_first'])
    h, dh = (d[n].double().reshape(b, count, hv, width, v.shape[-1]) for n in ('h', 'dh'))
    at = a.transpose(-1, -2)
    eg = torch.exp2(g)
    qgrad = do @ h.transpose(-1, -2) * eg * d['scale']
    kgrad = vn @ dh.transpose(-1, -2) * torch.exp2(g[..., -1:, :] - g)
    dw = -(dv @ h.transpose(-1, -2)).to(d['A'].dtype).double()
    kg = k * eg
    dkgb = at @ dw
    dvb = at @ dv
    dv2 = dvb * beta[..., None]
    db = (dvb * v).sum(-1) + (dkgb * kg).sum(-1)
    dg = q * qgrad - k * kgrad + kg * dkgb * beta[..., None]
    dg[..., -1, :] += (h * dh).sum(-1) * eg[..., -1, :] + (k * kgrad).sum(-2)
    dk = kgrad + dkgb * eg * beta[..., None]
    da = dv @ v.transpose(-1, -2) + dw @ kg.to(d['A'].dtype).double().transpose(-1, -2)
    da = torch.tril(da * beta[..., None, :], diagonal=-1)
    da = da.to(d['A'].dtype).double() @ at
    da = at @ da.to(d['A'].dtype).double()
    da = -torch.tril(da, diagonal=-1)
    errors = []
    for row, col in ((0, 0), (1, 1), (bt - 1, width - 1)):
        expected = math.fsum((float(do[0, 0, 0, row, z]) * float(h[0, 0, 0, col, z]) for z in range(v.shape[-1]))) * 2 ** float(g[0, 0, 0, row, col]) * d['scale']
        errors.append(abs(float(qgrad[0, 0, 0, row, col]) - expected))
        expected = math.fsum((float(at[0, 0, 0, row, j]) * float(dw[0, 0, 0, j, col]) for j in range(bt)))
        errors.append(abs(float(dkgb[0, 0, 0, row, col]) - expected))
    assert max(errors) < 1e-08, errors
    return ({name: builder.untile(value) for name, value in {'dq': qgrad, 'dk': dk, 'dv2': dv2, 'dg': dg, 'db': db[..., None], 'dA': da}.items()} | {'db': builder.untile(db[..., None]).squeeze(-1)}, {'method': 'CPU64 independent WY chain-rule matrix equations; original h layout, scale, beta and explicit fp16 intermediate casts; strictly lower dA inverse-gradient transform', 'scalar_max_abs_error': max(errors)})

def fwd_reference(d):
    bc, bt = (16, 64)
    q, k, g = (builder.tile(d[n], bc) for n in ('q', 'k', 'gk'))
    beta = builder.scalar_tile(d['beta'], bc)
    repeat = g.shape[2] // k.shape[2]
    q = q.repeat_interleave(repeat, dim=2)
    k = k.repeat_interleave(repeat, dim=2)
    gram = k * torch.exp2(g) @ (k * torch.exp2(-g)).transpose(-1, -2)
    lower = torch.tril(gram * beta[..., None], diagonal=-1)
    eye = torch.eye(bc, dtype=torch.float64).expand_as(lower)
    inverse = torch.linalg.solve_triangular(eye + lower, eye, upper=False, unitriangular=True)
    residual = float(((eye + lower) @ inverse - eye).abs().max())
    relative = float((((eye + lower) @ inverse - eye).abs() / (1 + (eye + lower).abs() @ inverse.abs())).max())
    assert relative < 1e-12, (residual, relative)
    qk = builder.untile(torch.tril(q * torch.exp2(g) @ (k * torch.exp2(-g)).transpose(-1, -2)) * 0.0884)
    b, t, heads, _ = qk.shape
    full = torch.zeros((b, t, heads, bt), dtype=torch.float64)
    for start in range(0, t, bc):
        col = start % bt
        full[:, start:start + bc, :, col:col + bc] = qk[:, start:start + bc]
    return ({'Aqk': full, 'Akkd': builder.untile(inverse)}, {'method': 'CPU64 causal 16x16 gated qk blocks and independent unit lower triangular solve for Akkd; Aqk off-diagonal blocks are not written by this kernel', 'triangular_solve_max_abs_residual': residual, 'triangular_solve_scaled_residual': relative, 'Aqk_defined_columns': 'same 16-column subblock as the token row within its original 64-token chunk'})

def reference(cid, d):
    return {['cv_5720b3845e530e2f', 'cv_f10c307064c4bac0', 'cv_ff9de30ff0adcd6a'][0]: intra_reference, ['cv_5720b3845e530e2f', 'cv_f10c307064c4bac0', 'cv_ff9de30ff0adcd6a'][1]: fused_reference, ['cv_5720b3845e530e2f', 'cv_f10c307064c4bac0', 'cv_ff9de30ff0adcd6a'][2]: fwd_reference}['cv_5720b3845e530e2f'](d)
