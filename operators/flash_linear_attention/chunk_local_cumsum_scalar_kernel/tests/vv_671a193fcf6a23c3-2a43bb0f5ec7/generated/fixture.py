"""Seeded CPU inputs and independent references, ported from the existing FLA test generator.
Logical fixture names are dictionary keys; this module reads and writes no data files.
Source SHA256: 146be15ebf8d7327e6769d15f01af2353d6d4b9978737cf529d245c31bed6336
Operation: chunk_local_cumsum_scalar
"""
def make_fixtures(seed=20261030):
    import torch
    from pathlib import PurePosixPath
    SEED = seed
    N = 65539
    selected = op = 'chunk_local_cumsum_scalar'
    fixtures = {}
    op = selected
    folder = PurePosixPath('') / op
    gen = torch.Generator(device='cpu').manual_seed(SEED)
    g = torch.randn((4, 2048, 8), generator=gen, dtype=torch.float32)
    result = torch.cat([part.cumsum(dim=1) for part in g.split(128, dim=1)], dim=1)
    for suffix, obj in [('_kernel_in.pt', dict(g=g, chunk_size=128)), ('_kernel_out.pt', dict(output=result))]:
        path = folder / (op + suffix)
        fixtures[path.name] = obj
    return fixtures
