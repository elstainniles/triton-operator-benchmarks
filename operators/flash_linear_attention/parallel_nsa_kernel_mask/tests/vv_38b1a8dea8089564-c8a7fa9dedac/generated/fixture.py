"""Seeded CPU inputs and independent references, ported from the existing FLA test generator.
Logical fixture names are dictionary keys; this module reads and writes no data files.
Source SHA256: 146be15ebf8d7327e6769d15f01af2353d6d4b9978737cf529d245c31bed6336
Operation: parallel_nsa_mask
"""
def make_fixtures(seed=20261030):
    import torch
    from pathlib import PurePosixPath
    SEED = seed
    N = 65539
    selected = op = 'parallel_nsa_mask'
    fixtures = {}
    folder = PurePosixPath('') / selected
    gen = torch.Generator(device='cpu').manual_seed(SEED)
    indices = torch.randint(0, 8, (1, 512, 4, 8), generator=gen, dtype=torch.int32)
    block = torch.arange(8, dtype=torch.int32)
    t_positions = torch.arange(512, dtype=torch.int32)
    out = (indices[..., None] == block).any(dim=-2) & (block[None, None, None, :] * 64 <= t_positions[None, :, None, None])
    for name, obj in [('block_indices.pt', indices), ('block_mask.pt', out)]:
        path = folder / name
        fixtures[path.name] = obj
    return fixtures
