"""Fixed-seed CPU inputs and independent references; no external tensor files."""
import torch

SEED = 20261030
BT = 128
SHAPE = (2, 257, 65)

def make_inputs():
    gen = torch.Generator(device='cpu').manual_seed(SEED)
    x = torch.randn(SHAPE, generator=gen, dtype=torch.float32)
    g = -0.03 - 0.17 * torch.rand(SHAPE, generator=gen, dtype=torch.float32)
    h0 = torch.randn((SHAPE[0], SHAPE[2]), generator=gen, dtype=torch.float32)
    return (x, g, h0)

def reference_closed_form(x, g, h0, bt=BT):
    """Evaluate each chunk via a triangular sum, not a sequential kernel replay."""
    x, g, h0 = (x.double(), g.double(), h0.double())
    B, T, D = x.shape
    out = torch.empty_like(x)
    cum = torch.empty_like(g)
    for start in range(0, T, bt):
        xx = x[:, start:start + bt]
        gg = g[:, start:start + bt]
        prefix = gg.cumsum(dim=1)
        L = xx.shape[1]
        weights = torch.exp(prefix[:, :, None, :] - prefix[:, None, :, :])
        lower = torch.ones((L, L), dtype=torch.bool).tril()
        weights = torch.where(lower[None, :, :, None], weights, 0.0)
        accumulated = (weights * xx[:, None, :, :]).sum(dim=2)
        if start == 0:
            accumulated += prefix.exp() * h0[:, None, :]
        out[:, start:start + L] = accumulated
        cum[:, start:start + L] = prefix
    return (out.float(), cum.float())

def make_fixtures():
    x, g, h0 = make_inputs()
    out, cum = reference_closed_form(x, g, h0)
    return {'x.pt': x, 'g.pt': g, 'h0.pt': h0, 'o.pt': out, 'gc.pt': cum}
