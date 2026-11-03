"""Seeded CPU inputs and independent references, ported from the existing FLA test generator.
Logical fixture names are dictionary keys; this module reads and writes no data files.
Source SHA256: 146be15ebf8d7327e6769d15f01af2353d6d4b9978737cf529d245c31bed6336
Operation: kda_gate_chunk_cumsum_vector
"""
def make_fixtures(seed=20261030):
    import torch
    from pathlib import PurePosixPath
    SEED = seed
    N = 65539
    selected = op = 'kda_gate_chunk_cumsum_vector'
    fixtures = {}
    folder = PurePosixPath('') / selected
    gen = torch.Generator(device='cpu').manual_seed(SEED)
    g = torch.randn((2, 1024, 8, 64), generator=gen, dtype=torch.float16)
    a = torch.randn((8,), generator=gen, dtype=torch.float16)
    bias = torch.randn((8 * 64,), generator=gen, dtype=torch.float16)
    gate = -a.double().exp()[None, None, :, None] * torch.nn.functional.softplus(g.double() + bias.double().view(8, 64))
    out = gate.reshape(2, 16, 64, 8, 64).cumsum(2).reshape(g.shape).float()
    for name, obj in [('g.pt', g), ('A_log.pt', a), ('dt_bias.pt', bias), ('output.pt', out)]:
        path = folder / name
        fixtures[path.name] = obj
    return fixtures
