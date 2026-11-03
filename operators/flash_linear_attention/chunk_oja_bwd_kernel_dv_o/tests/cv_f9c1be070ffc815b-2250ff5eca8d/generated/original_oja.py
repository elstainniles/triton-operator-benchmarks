"""Existing independent CPU equations; file-generation main omitted."""
import torch

def oja_reference(kind, payload):
    d = payload['input_data']
    x = {k: v.double() if isinstance(v, torch.Tensor) else v for k, v in d.items()}
    BT = d['BT']
    base = d['q'] if kind == 'oja_inter' else d['v']
    B, T, H, D = base.shape
    N = T // BT

    def b(k):
        return x[k].reshape(B, N, BT, H, -1).permute(0, 1, 3, 2, 4)

    def un(z):
        return z.permute(0, 1, 3, 2, 4).reshape(B, T, H, -1)
    if kind == 'oja_inter':
        q = (b('q') * d['scale']).float().double()
        return {'o': un(q @ x['h'] * b('gv').exp()), 'A': un((q @ b('k').transpose(-1, -2)).tril())}
    A = b('A').tril()
    g = b('gv' if kind == 'oja_intra' else 'g')
    v = b('v')
    out = torch.empty_like(v)
    for c in range(N):
        gain = (g[:, c, :, :, None, :] - g[:, c, :, None, :, :]).exp()
        weights = A[:, c, :, :, :, None] * gain
        if kind == 'oja_intra':
            out[:, c] = torch.einsum('bhijv,bhjv->bhiv', weights, v[:, c]) + b('o')[:, c]
        else:
            out[:, c] = torch.einsum('bhijv,bhiv->bhjv', weights, b('do')[:, c]) + b('dv')[:, c]
    if kind == 'oja_intra':
        return {'o': un(out)}
    return {'dv2': un(out), 'dg': x['o'] * x['do'] - x['v'] * un(out)}
