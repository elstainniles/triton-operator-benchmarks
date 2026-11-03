"""Existing independent CPU equations; file-generation main omitted."""
import torch

def as_tiles(value, bt):
    b, t, h = value.shape[:3]
    return value.double().reshape(b, t // bt, bt, h, *value.shape[3:]).transpose(2, 3)

def untiles(value):
    b, n, h, bt = value.shape[:4]
    return value.transpose(2, 3).reshape(b, n * bt, h, *value.shape[4:])

def reference(data, retain_casts=True):
    bt = data['BT']
    q, k, v, do = [as_tiles(data[n], bt) for n in ['q_star', 'k', 'v', 'do']]
    beta, g = [as_tiles(data[n], bt) for n in ['beta', 'g']]
    b, nt, h = q.shape[:3]
    state = data['h_kv'].double().reshape(b, nt, h, data['K'], data['V'])
    adj = data['dh_kv'].double().reshape_as(state)
    cast = lambda x, name: x.to(data[name].dtype).double() if retain_casts else x
    kb = cast(k * beta[..., None], 'k')
    decay = torch.tril(torch.exp(g[..., :, None] - g[..., None, :]))
    score = q @ kb.transpose(-1, -2) * decay
    score_grad = do @ v.transpose(-1, -2) * decay
    q_cross = do @ cast(state, 'do').transpose(-1, -2) * torch.exp(g)[..., None]
    k_cross = v @ cast(adj, 'v').transpose(-1, -2) * torch.exp(g[..., -1:] - g)[..., None]
    score_gate_grad = score * (do @ v.transpose(-1, -2))
    dg = score_gate_grad.sum(-1) - score_gate_grad.sum(-2)
    dg = dg - (k_cross * kb).sum(-1) + (q_cross * q).sum(-1)
    final_gate_grad = (state * adj).sum((-1, -2)) * torch.exp(g[..., -1]) + (k_cross * kb).sum((-1, -2))
    dg[..., -1] += final_gate_grad
    dq = q_cross + cast(score_grad, 'k') @ kb
    dk = k_cross + cast(score_grad, 'q_star').transpose(-1, -2) @ q
    dv = kb @ cast(adj, 'k') * torch.exp(g[..., -1:] - g)[..., None] + cast(score, 'do').transpose(-1, -2) @ do
    outputs = dict(dq=untiles(dq), dk=untiles(dk), dv=untiles(dv), dg=untiles(dg))
    return outputs
