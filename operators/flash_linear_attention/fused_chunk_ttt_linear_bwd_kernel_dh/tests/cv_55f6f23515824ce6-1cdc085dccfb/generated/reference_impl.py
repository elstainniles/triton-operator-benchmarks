"""Existing independent CPU equations; file-generation main omitted."""
import torch
import math

def dh_reference(d):
    """Adjoint recurrence using CPU matrix algebra on the supplied cached intermediates."""
    x = {k: v.double() if isinstance(v, torch.Tensor) else v for k, v in d.items()}
    B, T, H, K = d['q'].shape
    V = d['v'].shape[-1]
    BT = d['BT']
    N = math.ceil(T / BT)
    out = {k: torch.zeros_like(x[k]) for k in ('dk', 'dv', 'de', 'dw', 'db', 'dh0', 'dhb0')}
    dh = x['dht'].clone() if x['dht'] is not None else torch.zeros(B, H, K, V, dtype=torch.float64)
    dhb = x['dhbt'].reshape(B, H, V).clone() if x['dhbt'] is not None else torch.zeros(B, H, V, dtype=torch.float64)
    weight = x['w'][None, :, None, :]
    bias = x['b'][None, :, None, :]
    for i in range(N - 1, -1, -1):
        lo, hi = (i * BT, min((i + 1) * BT, T))

        def chunk(name):
            return x[name][:, lo:hi].transpose(1, 2)
        q, k, v, v2, xx, y, r, do = [chunk(n) for n in ('q', 'k', 'v', 'v2', 'x', 'y', 'r', 'do')]
        eta = chunk('eta').squeeze(-1)
        last = eta[..., -1]
        A = -torch.triu(k @ q.transpose(-1, -2) * d['scale'] * eta[..., None, :])
        Ae = -torch.triu(eta[..., None, :].expand(B, H, hi - lo, hi - lo))
        dvnew = A @ do + Ae @ do - last[..., None, None] * (k @ dh + dhb[..., None, :])
        dy = r * (dvnew - dvnew.mean(-1, keepdim=True) - xx * (dvnew * xx).mean(-1, keepdim=True))
        dx = -r * (dvnew * (xx * y).mean(-1, keepdim=True) + y * (dvnew * xx).mean(-1, keepdim=True))
        dr = (dvnew * v2 / r).sum(-1, keepdim=True)
        dv = -weight * dy
        dk = weight * dy
        out['dw'] += (2 * weight * xx * dy + (bias - v + k) * dy).sum(-2)
        out['db'] += (weight * dy).sum(-2)
        dx = dx + weight.square() * dy
        dkh = r * (dx - dx.mean(-1, keepdim=True) - xx * (xx * dx).mean(-1, keepdim=True)) - r.square() * dr * xx / V
        hist = x['h'][:, i]
        dk += dkh @ hist.transpose(-1, -2)
        ds = torch.tril(do @ v2.transpose(-1, -2))
        qs = q * d['scale']
        dk -= last[..., None, None] * (v2 @ dh.transpose(-1, -2))
        dk -= ds.transpose(-1, -2) @ (qs * eta[..., None])
        de = -(ds @ k * qs).sum(-1) - ds.sum(-1)
        de[..., -1] -= (dh * (k.transpose(-1, -2) @ v2)).sum((-1, -2)) + (dhb * v2.sum(-2)).sum(-1)
        dh = dh + qs.transpose(-1, -2) @ do + k.transpose(-1, -2) @ dkh
        dhb = dhb + (do + dkh).sum(-2)
        out['dk'][:, lo:hi] = dk.transpose(1, 2)
        out['dv'][:, lo:hi] = dv.transpose(1, 2)
        out['de'][:, lo:hi] = de.transpose(1, 2).unsqueeze(-1)
    out['dh0'] = dh
    out['dhb0'] = dhb.reshape_as(x['dhb0'])
    out['dw_sum'] = out['dw'].sum(0)
    out['db_sum'] = out['db'].sum(0)
    return out

def check_dh_autograd():
    """Independent differentiable forward TTT model verifies every adjoint output."""
    gen = torch.Generator().manual_seed(20261031)
    B, T, H, D, BT = (1, 6, 1, 4, 2)

    def variable(*shape):
        return (torch.randn(shape, generator=gen, dtype=torch.float64) * 0.1).requires_grad_()
    q, k, v = (variable(B, T, H, D), variable(B, T, H, D), variable(B, T, H, D))
    w, b = (variable(H, D), variable(H, D))
    eta = variable(B, T, H, 1)
    h0, hb0 = (variable(B, H, D, D), variable(B, H, 1, D))
    do = variable(B, T, H, D).detach()
    dht = variable(B, H, D, D).detach()
    dhbt = variable(B, H, 1, D).detach()
    h, hb = (h0[0, 0], hb0[0, 0, 0])
    outputs = []
    cache = {n: [] for n in ('h', 'v2', 'x', 'y', 'r')}
    for lo in range(0, T, BT):
        kk, vv, qq = (k[0, lo:lo + BT, 0], v[0, lo:lo + BT, 0], q[0, lo:lo + BT, 0])
        z = kk @ h + hb
        norm = torch.nn.functional.layer_norm(z, (D,), eps=1e-05)
        residual = norm * w[0] + b[0] - vv + kk
        update = torch.autograd.grad(0.5 * residual.square().sum(), z, create_graph=True)[0]
        causal = torch.tril(qq @ kk.T * 0.5 + torch.ones(BT, BT, dtype=torch.float64))
        e = eta[0, lo:lo + BT, 0, 0]
        outputs.append(qq @ h * 0.5 + hb - e[:, None] * causal @ update)
        cache['h'].append(h.detach())
        cache['v2'].append(update.detach())
        cache['x'].append(norm.detach())
        cache['y'].append((residual * w[0]).detach())
        cache['r'].append((z.var(-1, keepdim=True, unbiased=False) + 1e-05).rsqrt().detach())
        h = h - e[-1] * (kk.T @ update)
        hb = hb - e[-1] * update.sum(0)
    loss = (torch.cat(outputs, 0) * do[0, :, 0]).sum() + (h * dht[0, 0]).sum() + (hb * dhbt[0, 0, 0]).sum()
    names = ('dk', 'dv', 'de', 'dw_sum', 'db_sum', 'dh0', 'dhb0')
    grads = torch.autograd.grad(loss, (k, v, eta, w, b, h0, hb0))
    data = {n: t.detach() for n, t in {'q': q, 'k': k, 'v': v, 'w': w, 'b': b, 'eta': eta, 'do': do, 'dht': dht, 'dhbt': dhbt}.items()}
    data.update(h=torch.stack(cache['h']).reshape(B, T // BT, H, D, D), **{n: torch.cat(cache[n], 0).reshape(B, T, H, -1) for n in ('v2', 'x', 'y', 'r')}, dh0=torch.zeros_like(h0), dhb0=torch.zeros_like(hb0), dk=torch.zeros_like(k), dv=torch.zeros_like(v), de=torch.zeros_like(eta), dw=torch.zeros(B, H, D, dtype=torch.float64), db=torch.zeros(B, H, D, dtype=torch.float64), BT=BT, scale=0.5)
    actual = dh_reference(data)
    errors = {}
    for name, gold in zip(names, grads):
        torch.testing.assert_close(actual[name], gold, rtol=1e-09, atol=1e-10)
        errors[name] = float((actual[name] - gold).abs().max())
    return errors

def reference(cid, d):
    return (dh_reference(d), {'method': 'Independent CPU64 adjoint matrix recurrence on original supplied cached intermediates', 'forward_autograd_gradient_checks': check_dh_autograd()})
