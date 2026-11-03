"""Seeded CPU inputs and independent references, ported from the existing FLA test generator.
Logical fixture names are dictionary keys; this module reads and writes no data files.
Source SHA256: 146be15ebf8d7327e6769d15f01af2353d6d4b9978737cf529d245c31bed6336
Operation: prepare_position_ids
"""
def make_fixtures(seed=20261030):
    import torch
    from pathlib import PurePosixPath
    SEED = seed
    N = 65539
    selected = op = 'prepare_position_ids'
    fixtures = {}
    folder = PurePosixPath('') / 'prepare_position_ids'
    gen = torch.Generator(device='cpu').manual_seed(SEED)
    lengths = torch.randint(1, 1024, (32,), generator=gen, dtype=torch.int32)
    cu = torch.cat((torch.zeros(1, dtype=torch.int32), lengths.cumsum(0, dtype=torch.int32)))
    ref = torch.cat([torch.arange(n, dtype=torch.int32) for n in lengths.tolist()])
    for name, obj in [('prepare_position_ids_kernel_in.pt', dict(y=torch.zeros_like(ref), cu_seqlens=cu, B_KERNEL=128, B=32)), ('prepare_position_ids_kernel_out.pt', dict(output=ref))]:
        path = folder / name
        fixtures[path.name] = obj
    return fixtures
