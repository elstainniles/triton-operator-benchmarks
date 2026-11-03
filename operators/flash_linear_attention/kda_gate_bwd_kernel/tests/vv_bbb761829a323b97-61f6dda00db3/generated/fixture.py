"""Fixed-seed CPU inputs and independent references; no external tensor files."""
import torch
import torch.nn.functional as F

SEED = 20261030
SHAPE = (2, 1024, 8, 64)
FILES = ('g.pt', 'A_log.pt', 'dt_bias.pt', 'dyg.pt', 'dg.pt', 'dA.pt', 'dbias.pt')

def make_inputs():
    gen = torch.Generator(device='cpu').manual_seed(SEED)
    b, t, h, k = SHAPE
    g = torch.randn((b, t, h, k), generator=gen, dtype=torch.float16)
    a = torch.randn((h,), generator=gen, dtype=torch.float16)
    bias = torch.randn((h * k,), generator=gen, dtype=torch.float16)
    dy = torch.randn((b, t, h, k), generator=gen, dtype=torch.float16)
    return (g, a, bias, dy)

def reference_autograd(g, a, bias, dy):
    """Differentiate the real-valued forward formula; apply wrapper output casts."""
    h, k = g.shape[-2:]
    g64 = g.double().requires_grad_(True)
    a64 = a.double().requires_grad_(True)
    bias64 = bias.double().requires_grad_(True)
    y = -a64.exp().view(h, 1) * F.softplus(g64 + bias64.view(h, k))
    dg64, da64, dbias64 = torch.autograd.grad(y, (g64, a64, bias64), dy.double())
    dg = dg64.float().to(g.dtype)
    da = da64.float().to(a.dtype)
    dbias = dg.reshape(-1, h * k).float().sum(0).to(bias.dtype)
    return (dg, da, dbias, dbias64)

def make_fixtures():
    g, a, bias, dy = make_inputs()
    dg, da, dbias, _ = reference_autograd(g, a, bias, dy)
    return dict(zip(FILES, (g, a, bias, dy, dg, da, dbias), strict=True))
