"""Seeded CPU inputs and independent references, ported from the existing FLA test generator.
Logical fixture names are dictionary keys; this module reads and writes no data files.
Source SHA256: 146be15ebf8d7327e6769d15f01af2353d6d4b9978737cf529d245c31bed6336
Operation: layer_norm_gated_bwd1
"""
def make_fixtures(seed=20261030):
    import torch
    from pathlib import PurePosixPath
    SEED = seed
    N = 65539
    selected = op = 'layer_norm_gated_bwd1'
    fixtures = {}
    folder = PurePosixPath('') / selected
    gen = torch.Generator(device='cpu').manual_seed(SEED)
    dim = 256 if selected.endswith('bwd') else 1024
    x = torch.randn((2048, dim), generator=gen) * 0.1
    g = torch.randn((2048, dim), generator=gen) * 0.1
    w = torch.randn((dim,), generator=gen)
    b = torch.randn((dim,), generator=gen)
    dy = torch.randn((2048, dim), generator=gen) * 0.1
    xx = x.double().requires_grad_(True)
    gg = g.double().requires_grad_(True)
    ww = w.double().requires_grad_(True)
    bb = b.double().requires_grad_(True)
    mean = xx.detach().mean(-1).float()
    rstd = (xx.detach().var(-1, unbiased=False) + 1e-05).rsqrt().float()
    yy = torch.nn.functional.layer_norm(xx, (dim,), ww, bb, 1e-05) * torch.nn.functional.silu(gg)
    dx, dg, dw, db = torch.autograd.grad(yy, (xx, gg, ww, bb), dy.double())
    name = 'layer_norm_gated_bwd_kernel' if dim == 256 else 'layer_norm_gated_bwd_kernel1'
    path = folder / ('test_' + name + '_cuda.pt')
    fixtures[path.name] = dict(input_data=dict(x=x, g=g, weight=w, bias=b, dy=dy, mean=mean, rstd=rstd, T=2048, D=dim, activation='swish', eps=1e-05, is_rms_norm=False), gpu_output=dict(dx=dx.float(), dg=dg.float(), dw=dw.float(), db=db.float()))
    return fixtures
