"""Seeded CPU inputs and independent references, ported from the existing FLA test generator.
Logical fixture names are dictionary keys; this module reads and writes no data files.
Source SHA256: 146be15ebf8d7327e6769d15f01af2353d6d4b9978737cf529d245c31bed6336
Operation: softmax_fwd
"""
def make_fixtures(seed=20261030):
    import torch
    from pathlib import PurePosixPath
    SEED = seed
    N = 65539
    selected = op = 'softmax_fwd'
    fixtures = {}
    folder = PurePosixPath('') / op
    gen = torch.Generator(device='cpu').manual_seed(SEED)
    x = torch.randn((1024, 256), generator=gen, dtype=torch.float32)
    inp = dict(x=x)
    out = dict(output=torch.softmax(x, dim=-1))
    for suffix, obj in [('_kernel_in.pt', inp), ('_kernel_out.pt', out)]:
        path = folder / (op + suffix)
        fixtures[path.name] = obj
    return fixtures
