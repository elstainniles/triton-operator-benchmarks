"""Seeded CPU inputs and independent references, ported from the existing FLA test generator.
Logical fixture names are dictionary keys; this module reads and writes no data files.
Source SHA256: 146be15ebf8d7327e6769d15f01af2353d6d4b9978737cf529d245c31bed6336
Operation: rotary_embedding
"""
def make_fixtures(seed=20261030):
    import torch
    from pathlib import PurePosixPath
    SEED = seed
    N = 65539
    selected = op = 'rotary_embedding'
    fixtures = {}
    folder = PurePosixPath('') / selected
    gen = torch.Generator(device='cpu').manual_seed(SEED)
    x = torch.randn((2, 4096, 4, 128), generator=gen, dtype=torch.float32)
    inv_freq = 1 / 10000.0 ** (torch.arange(0, 128, 2, dtype=torch.float32) / 128)
    theta = torch.outer(torch.arange(4096, dtype=torch.float32), inv_freq)
    cos, sin = (theta.cos(), theta.sin())
    x0, x1 = x.chunk(2, dim=-1)
    c, s = (cos[None, :, None, :], sin[None, :, None, :])
    y = torch.cat((x0 * c - x1 * s, x0 * s + x1 * c), dim=-1)
    for name, obj in [('rotary_embedding_kernel_in.pt', dict(x=x, cos=cos, sin=sin)), ('rotary_embedding_kernel_out.pt', dict(y=y))]:
        path = folder / name
        fixtures[path.name] = obj
    return fixtures
