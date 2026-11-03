"""Seeded CPU inputs and independent references, ported from the existing FLA test generator.
Logical fixture names are dictionary keys; this module reads and writes no data files.
Source SHA256: 146be15ebf8d7327e6769d15f01af2353d6d4b9978737cf529d245c31bed6336
Operation: l2norm_bwd
"""
def make_fixtures(seed=20261030):
    import torch
    from pathlib import PurePosixPath
    SEED = seed
    N = 65539
    selected = op = 'l2norm_bwd'
    fixtures = {}
    folder = PurePosixPath('') / selected
    gen = torch.Generator(device='cpu').manual_seed(SEED)
    dim = 256 if selected == 'l2norm_bwd' else 1024
    x = torch.randn((4, 2048, dim), generator=gen, dtype=torch.float32)
    dy = torch.randn(x.shape, generator=gen, dtype=torch.float32)
    xx = x.double().detach().requires_grad_(True)
    rr = torch.rsqrt(xx.square().sum(-1) + 1e-06)
    yy = xx * rr[..., None]
    dx, = torch.autograd.grad(yy, xx, dy.double())
    for suffix, obj in [('_in.pt', dict(x=x, y=yy.float(), rstd=rr.float(), dy=dy)), ('_out.pt', dict(dx=dx.float()))]:
        path = folder / (('l2norm_bwd_kernel' if selected == 'l2norm_bwd' else 'l2norm_bwd_kernel1') + suffix)
        fixtures[path.name] = obj
    return fixtures
