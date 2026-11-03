"""Seeded CPU inputs and independent references, ported from the existing FLA test generator.
Logical fixture names are dictionary keys; this module reads and writes no data files.
Source SHA256: 146be15ebf8d7327e6769d15f01af2353d6d4b9978737cf529d245c31bed6336
Operation: compare_and_swap
"""
def make_fixtures(seed=20261030):
    import torch
    from pathlib import PurePosixPath
    SEED = seed
    N = 65539
    selected = op = 'compare_and_swap'
    fixtures = {}
    folder = PurePosixPath('') / selected
    gen = torch.Generator(device='cpu').manual_seed(SEED)
    x = torch.randperm(4096, generator=gen).float() / 4096
    ids = torch.randperm(4096, generator=gen).int()
    ox = x.clone()
    oi = ids.clone()
    stride = 1 if selected == 'bitonic_merge' else 2048
    for pos in range(4096):
        partner = pos ^ stride
        left = pos & ~stride
        right = left + stride
        if x[left] > x[right]:
            ox[pos] = x[partner]
            oi[pos] = ids[partner]
    for name, obj in [('input_x.pt', x), ('input_ids.pt', ids), ('output_x.pt', ox), ('output_ids.pt', oi)]:
        path = folder / name
        fixtures[path.name] = obj
    return fixtures
