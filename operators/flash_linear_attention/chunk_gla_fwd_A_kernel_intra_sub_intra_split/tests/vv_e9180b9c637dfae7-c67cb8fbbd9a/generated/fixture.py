"""Seeded CPU inputs and independent references, ported from the existing FLA test generator.
Logical fixture names are dictionary keys; this module reads and writes no data files.
Source SHA256: 146be15ebf8d7327e6769d15f01af2353d6d4b9978737cf529d245c31bed6336
Operation: chunk_gla_intra_split
"""
def make_fixtures(seed=20261030):
    import torch
    from pathlib import PurePosixPath
    SEED = seed
    N = 65539
    selected = op = 'chunk_gla_intra_split'
    fixtures = {}
    folder = PurePosixPath('') / selected
    gen = torch.Generator(device='cpu').manual_seed(SEED)
    shape = (2, 2048, 8, 512)
    q = torch.randn(shape, generator=gen, dtype=torch.float16)
    k = torch.randn(shape, generator=gen, dtype=torch.float16)
    g = torch.randn(shape, generator=gen, dtype=torch.float16)
    out = torch.empty((4, 2, 2048, 8, 16), dtype=torch.float32)
    scale = 512 ** (-0.5)
    for part in range(4):
        sel = slice(part * 128, (part + 1) * 128)
        qq = q[..., sel].double().reshape(2, 128, 16, 8, 128).permute(0, 1, 3, 2, 4)
        kk = k[..., sel].double().reshape(2, 128, 16, 8, 128).permute(0, 1, 3, 2, 4)
        gg = g[..., sel].double().reshape(2, 128, 16, 8, 128).permute(0, 1, 3, 2, 4)
        vals = torch.matmul(qq * torch.exp2(gg), (kk * torch.exp2(-gg)).transpose(-1, -2)) * scale
        out[part] = torch.tril(vals).permute(0, 1, 3, 2, 4).reshape(2, 2048, 8, 16).float()
    for name, obj in [('q.pt', q), ('k.pt', k), ('g.pt', g), ('A_intra.pt', out)]:
        path = folder / name
        fixtures[path.name] = obj
    return fixtures
