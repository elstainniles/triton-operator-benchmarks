"""Existing independent CPU equations; file-generation main omitted."""
import torch

def reference(kind, d, alt=False):
    x = {k: v.double() if isinstance(v, torch.Tensor) else v for k, v in d.items()}

    def mm(a, b):
        return torch.einsum('...ik,...kj->...ij', a, b) if alt else a @ b
    base = d['kg']
    B, T, H, K = base.shape
    BT = d['chunk_size']
    N = T // BT

    def b(key):
        return x[key].reshape(B, N, BT, H, -1).permute(0, 1, 3, 2, 4)

    def un(z):
        return z.permute(0, 1, 3, 2, 4).reshape(B, T, H, -1)
    return {'dv': un(mm(b('kg'), x['dh']) + mm(b('A_qk').tril().transpose(-1, -2), b('do')))}
