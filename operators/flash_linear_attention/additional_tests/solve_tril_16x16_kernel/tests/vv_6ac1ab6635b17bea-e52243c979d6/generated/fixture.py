"""Seeded CPU inputs and independent references, ported from the existing FLA test generator.
Logical fixture names are dictionary keys; this module reads and writes no data files.
Source SHA256: 146be15ebf8d7327e6769d15f01af2353d6d4b9978737cf529d245c31bed6336
Operation: solve_tril_16x16
"""
def make_fixtures(seed=20261030):
    import torch
    from pathlib import PurePosixPath
    SEED = seed
    N = 65539
    selected = op = 'solve_tril_16x16'
    fixtures = {}
    folder = PurePosixPath('') / selected
    gen = torch.Generator(device='cpu').manual_seed(SEED)
    k = torch.nn.functional.normalize(torch.randn((4, 4, 256, 64), generator=gen, dtype=torch.float32), dim=-1)
    blocks = k.reshape(4, 4, 16, 16, 64)
    a = (blocks @ blocks.transpose(-1, -2)).tril(-1)
    matrix = a.double() + torch.eye(16, dtype=torch.float64)
    inv = torch.linalg.solve_triangular(matrix, torch.eye(16, dtype=torch.float64).expand_as(matrix), upper=False, unitriangular=True).float()
    a_input = a.reshape(4, 4, 256, 16).transpose(1, 2).contiguous()
    output = inv.reshape(4, 4, 256, 16).transpose(1, 2).contiguous()
    for suffix, obj in [('_in.pt', dict(A_input=a_input, B=4, T=256, H=4, BT=16)), ('_out.pt', dict(output=output))]:
        path = folder / ('solve_tril_16x16_kernel' + suffix)
        fixtures[path.name] = obj
    return fixtures
