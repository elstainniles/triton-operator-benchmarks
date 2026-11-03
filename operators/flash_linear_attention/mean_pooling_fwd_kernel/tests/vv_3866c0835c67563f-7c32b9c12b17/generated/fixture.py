"""Seeded CPU inputs and independent references, ported from the existing FLA test generator.
Logical fixture names are dictionary keys; this module reads and writes no data files.
Source SHA256: 146be15ebf8d7327e6769d15f01af2353d6d4b9978737cf529d245c31bed6336
Operation: mean_pooling_fwd
"""
def make_fixtures(seed=20261030):
    import torch
    from pathlib import PurePosixPath
    SEED = seed
    N = 65539
    selected = op = 'mean_pooling_fwd'
    fixtures = {}
    folder = PurePosixPath('') / op
    gen = torch.Generator(device='cpu').manual_seed(SEED)
    shape = (2, 257, 4, 64)
    chunk = 128
    x = torch.randn(shape, generator=gen, dtype=torch.float32)
    result = torch.stack([z.mean(dim=1) for z in x.split(chunk, dim=1)], dim=1)
    inp = dict(x=x, chunk_size=chunk)
    for name, obj in [(op + '_kernel_in.pt', inp), (op + '_kernel_out.pt', dict(output=result))]:
        path = folder / name
        fixtures[path.name] = obj
    return fixtures
