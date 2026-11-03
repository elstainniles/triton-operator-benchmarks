"""Seeded CPU inputs and independent references, ported from the existing FLA test generator.
Logical fixture names are dictionary keys; this module reads and writes no data files.
Source SHA256: 146be15ebf8d7327e6769d15f01af2353d6d4b9978737cf529d245c31bed6336
Operation: l2norm_fwd1
"""
def make_fixtures(seed=20261030):
    import torch
    from pathlib import PurePosixPath
    SEED = seed
    N = 65539
    selected = op = 'l2norm_fwd1'
    fixtures = {}
    folder = PurePosixPath('') / selected
    gen = torch.Generator(device='cpu').manual_seed(SEED)
    dim = 256 if selected == 'l2norm_fwd' else 1024
    x = torch.randn((4, 2048, dim), generator=gen, dtype=torch.float32)
    rr = torch.rsqrt(x.double().square().sum(-1) + 1e-06)
    y, rstd = ((x.double() * rr[..., None]).float(), rr.float())
    filename = 'l2norm_fwd_kernel' if selected == 'l2norm_fwd' else 'l2norm_fwd_kernel1'
    for suffix, obj in [('_in.pt', dict(x=x)), ('_out.pt', dict(y=y, rstd=rstd))]:
        path = folder / (filename + suffix)
        fixtures[path.name] = obj
    return fixtures
