"""Seeded CPU inputs and independent references, ported from the existing FLA test generator.
Logical fixture names are dictionary keys; this module reads and writes no data files.
Source SHA256: 146be15ebf8d7327e6769d15f01af2353d6d4b9978737cf529d245c31bed6336
Operation: layer_norm_gated_fwd
"""
def make_fixtures(seed=20261030):
    import torch
    from pathlib import PurePosixPath
    SEED = seed
    N = 65539
    selected = op = 'layer_norm_gated_fwd'
    fixtures = {}
    folder = PurePosixPath('') / selected
    gen = torch.Generator(device='cpu').manual_seed(SEED)
    dim = 256 if selected.endswith('fwd') else 1200
    x = torch.randn((8192, dim), generator=gen)
    g = torch.randn((8192, dim), generator=gen)
    w = torch.randn((dim,), generator=gen)
    b = torch.randn((dim,), generator=gen)
    xx = x.double()
    gg = g.double()
    mean = xx.mean(-1)
    rstd = (xx.var(-1, unbiased=False) + 1e-05).rsqrt()
    y = ((xx - mean[:, None]) * rstd[:, None] * w.double() + b.double()) * torch.nn.functional.silu(gg)
    name = 'layer_norm_gated_fwd_kernel' if dim == 256 else 'layer_norm_gated_fwd_kernel1'
    for suffix, obj in [('_in.pt', dict(x=x, g=g, w=w, b=b)), ('_out.pt', dict(y=y.float(), mean=mean.float(), rstd=rstd.float(), residual_out=x))]:
        path = folder / (name + suffix)
        fixtures[path.name] = obj
    return fixtures
