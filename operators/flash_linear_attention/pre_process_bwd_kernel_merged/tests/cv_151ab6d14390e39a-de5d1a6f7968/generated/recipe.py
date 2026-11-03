"""Existing independent CPU equations; file-generation main omitted."""
import torch

def make_fixtures():
    gen = torch.Generator(device='cpu').manual_seed(SEED)
    rand = lambda shape: torch.randn(shape, generator=gen, dtype=torch.float32) * 0.05
    v = dict(q=rand((1, 128, 2, 64)), k=rand((1, 128, 2, 64)), w=rand((1, 128, 4, 64)), do=rand((1, 128, 4, 32)), dv=rand((1, 128, 4, 32)), g=-torch.rand((1, 128, 4), generator=gen, dtype=torch.float32) * 0.5, gk=None, bg=None, scale=0.125, cu_seqlens=None, chunk_size=64)
    y = affine_reference(v)
    torch.testing.assert_close(y, recurrence_reference(v), atol=1e-05, rtol=1e-05)
    return {'pre_process_bwd_seed20261030.pt': {'input_data': v, 'gpu_output': {'dhm': y}}}
SEED = 20261030

def affine_reference(v):
    result = []
    for h in range(4):
        transfers = []
        for begin in range(0, 128, 64):
            sl = slice(begin, begin + 64)
            q, k = [v[n][0, sl, h // 2].double() for n in ('q', 'k')]
            w, do, dv = [v[n][0, sl, h].double() for n in ('w', 'do', 'dv')]
            g = v['g'][0, sl, h].double()
            A = torch.exp2(g[-1]) * torch.eye(64, dtype=torch.float64) - w.T @ (torch.exp2(g[-1] - g)[:, None] * k)
            b = v['scale'] * (q * torch.exp2(g)[:, None]).T @ do - w.T @ dv
            transfers.append((A, b))
        A0, b0 = transfers[0]
        A1, b1 = transfers[1]
        result.append(torch.cat((A0 @ b1 + b0, A0 @ A1), dim=1))
    return torch.stack(result).float()

def recurrence_reference(v):
    result = []
    for h in range(4):
        dh = torch.zeros((64, 32), dtype=torch.float64)
        dm = torch.eye(64, dtype=torch.float64)
        for begin in (64, 0):
            sl = slice(begin, begin + 64)
            q, k = [v[n][0, sl, h // 2].double() for n in ('q', 'k')]
            w, do, dv = [v[n][0, sl, h].double() for n in ('w', 'do', 'dv')]
            g = v['g'][0, sl, h].double()
            gate = torch.exp2(g[-1] - g)
            dvalue = gate[:, None] * (k @ dh) + dv
            dh = torch.exp2(g[-1]) * dh + v['scale'] * (q * torch.exp2(g)[:, None]).T @ do - w.T @ dvalue
            dm = (torch.exp2(g[-1]) * torch.eye(64, dtype=torch.float64) - w.T @ (gate[:, None] * k)) @ dm
        result.append(torch.cat((dh, dm), dim=1))
    return torch.stack(result).float()
