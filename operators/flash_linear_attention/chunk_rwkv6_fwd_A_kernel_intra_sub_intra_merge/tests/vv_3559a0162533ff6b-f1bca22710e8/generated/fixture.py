"""Seeded CPU inputs and independent references, ported from the existing FLA test generator.
Logical fixture names are dictionary keys; this module reads and writes no data files.
Source SHA256: 146be15ebf8d7327e6769d15f01af2353d6d4b9978737cf529d245c31bed6336
Operation: chunk_rwkv6_intra_merge
"""
def make_fixtures(seed=20261030):
    import torch
    from pathlib import PurePosixPath
    SEED = seed
    N = 65539
    selected = op = 'chunk_rwkv6_intra_merge'
    fixtures = {}
    folder = PurePosixPath('') / selected
    gen = torch.Generator(device='cpu').manual_seed(SEED)
    local = torch.randn((4, 4, 1000, 4, 16), generator=gen) * 0.1
    initial = torch.randn((4, 1000, 4, 64), generator=gen) * 0.1
    gi = torch.randn((4, 1000, 4, 512), generator=gen) * 0.1
    ge = torch.randn((4, 1000, 4, 512), generator=gen) * 0.1
    u = torch.randn((4, 512), generator=gen, dtype=torch.float16)
    out = initial.clone()
    vals = local.double().sum(0).float()
    for t in range(0, 1000, 16):
        n = min(16, 1000 - t)
        slot = t % 64
        out[:, t:t + n, :, slot:slot + 16] = vals[:, t:t + n]
    inp = dict(A_intra=local, A=initial, gi=gi, ge=ge, u=u, scale=512 ** (-0.5), chunk_size=64, B=4, T=1000, H=4, BT=64, BC=16, NK=4, NT=16, NC=4)
    for name, obj in [('inputs.pt', inp), ('outputs.pt', dict(A_out=out)), ('outputs_fp32.pt', dict(A_out_fp32=out))]:
        path = folder / name
        fixtures[path.name] = obj
    return fixtures
