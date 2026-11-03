"""Existing independent CPU equations; file-generation main omitted."""
import torch
import math

def reference(inputs, chunk_size=32):
    q, k, v, g, do = (inputs[n].double() for n in ('q', 'k', 'v', 'g', 'do'))
    B, T, H, K = q.shape
    V = v.shape[-1]
    NT = T // chunk_size
    if T % chunk_size:
        raise ValueError('this frozen original input uses complete chunks')

    def chunks(x):
        return x.reshape(B, NT, chunk_size, H, x.shape[-1]).permute(0, 1, 3, 2, 4)
    qc = chunks(q).detach().requires_grad_(True)
    kc = chunks(k).detach().requires_grad_(True)
    vc, doc = (chunks(v), chunks(do))
    gc = chunks(g) * math.log(2.0)
    raw = torch.cat((gc[..., :1, :], gc[..., 1:, :] - gc[..., :-1, :]), dim=-2).detach().requires_grad_(True)
    cumulative = raw.cumsum(-2)
    h = inputs['h'].double().reshape(B, NT, H, K, V)
    dh = inputs['dh'].double().reshape(B, NT, H, K, V)
    intra_q = chunks(inputs['dq'].double())
    intra_k = chunks(inputs['dk'].double())
    attention = torch.matmul(qc * cumulative.exp(), h) * inputs['scale']
    final_state = cumulative[..., -1, :].exp()[..., None] * h
    weighted_keys = kc * (cumulative[..., -1:, :] - cumulative).exp()
    final_state = final_state + torch.matmul(weighted_keys.transpose(-1, -2), vc)
    local = (intra_q * qc * (cumulative - gc).exp() + intra_k * kc * (gc - cumulative).exp()).sum()
    objective = (attention * doc).sum() + (final_state * dh).sum() + local
    dq, dk, dg = torch.autograd.grad(objective, (qc, kc, raw))

    def restored(x):
        return x.permute(0, 1, 3, 2, 4).reshape(B, T, H, K).contiguous()
    return {n: restored(x) for n, x in zip(('dq2', 'dk2', 'dg'), (dq, dk, dg))}
