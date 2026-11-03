"""Seeded CPU inputs and independent references, ported from the existing FLA test generator.
Logical fixture names are dictionary keys; this module reads and writes no data files.
Source SHA256: 146be15ebf8d7327e6769d15f01af2353d6d4b9978737cf529d245c31bed6336
Operation: chunk_gla_intra_merge
"""
def make_fixtures(seed=20261030):
    import torch
    from pathlib import PurePosixPath
    SEED = seed
    N = 65539
    selected = op = 'chunk_gla_intra_merge'
    fixtures = {}
    folder = PurePosixPath('') / selected
    gen = torch.Generator(device='cpu').manual_seed(SEED)
    local = torch.randn((4, 2, 1024, 8, 16), generator=gen) * 0.1
    initial = torch.randn((2, 1024, 8, 64), generator=gen) * 0.1
    values = local.double().sum(0).float()
    out = initial.clone()
    for t in range(0, 1024, 16):
        slot = t % 64
        out[:, t:t + 16, :, slot:slot + 16] = values[:, t:t + 16]
    inp = dict(A_intra=local, A=initial, T=1024, B=2, H=8, BT=64, BC=16, BK=128, NK=4, NT=16, NC=4)
    for name, obj in [('inputs.pt', inp), ('outputs.pt', dict(A=out)), ('outputs_fp32.pt', dict(A_fp32=out))]:
        path = folder / name
        fixtures[path.name] = obj
    return fixtures
