"""Existing independent CPU equations; file-generation main omitted."""
import torch

def algebraic_control(inputs, chunk_size=32):
    q, k, v, g, do = (inputs[n].double() for n in ('q', 'k', 'v', 'g', 'do'))
    B, T, H, K = q.shape
    V = v.shape[-1]
    NT = T // chunk_size

    def chunks(x):
        return x.reshape(B, NT, chunk_size, H, x.shape[-1]).permute(0, 1, 3, 2, 4)
    q, k, v, g, do = map(chunks, (q, k, v, g, do))
    h = inputs['h'].double().reshape(B, NT, H, K, V)
    dh = inputs['dh'].double().reshape(B, NT, H, K, V)
    dq = torch.matmul(do, h.transpose(-1, -2)) * inputs['scale'] * torch.exp2(g)
    dk_inter = torch.matmul(v, dh.transpose(-1, -2)) * torch.exp2(g[..., -1:, :] - g)
    edge = (h * dh).sum(-1) * torch.exp2(g[..., -1, :]) + (dk_inter * k).sum(-2)
    dq = dq + chunks(inputs['dq'].double())
    dk = dk_inter + chunks(inputs['dk'].double())
    products = q * dq - k * dk
    dg = products.flip(-2).cumsum(-2).flip(-2) + edge[..., None, :]

    def restored(x):
        return x.permute(0, 1, 3, 2, 4).reshape(B, T, H, K).contiguous()
    return {n: restored(x) for n, x in zip(('dq2', 'dk2', 'dg'), (dq, dk, dg))}
