"""Seeded CPU inputs and independent references, ported from the existing FLA test generator.
Logical fixture names are dictionary keys; this module reads and writes no data files.
Source SHA256: 146be15ebf8d7327e6769d15f01af2353d6d4b9978737cf529d245c31bed6336
Operation: save_intra_chunk_attn
"""
def make_fixtures(seed=20261030):
    import torch
    from pathlib import PurePosixPath
    SEED = seed
    N = 65539
    selected = op = 'save_intra_chunk_attn'
    fixtures = {}
    folder = PurePosixPath('') / selected
    gen = torch.Generator(device='cpu').manual_seed(SEED)
    local = torch.randn((4, 8, 2048, 32), generator=gen, dtype=torch.float16)
    initial = torch.zeros((4, 8, 2048, 2048), dtype=torch.float16)
    out = initial.clone()
    for t in range(0, 2048, 32):
        out[:, :, t:t + 32, t:t + 32] = local[:, :, t:t + 32]
    for name, obj in [('inputs.pt', dict(A_local=local, A=initial, T=2048, BT=32)), ('outputs.pt', dict(A=out))]:
        path = folder / name
        fixtures[path.name] = obj
    return fixtures
