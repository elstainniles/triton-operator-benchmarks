"""Seeded CPU inputs and independent references, ported from the existing FLA test generator.
Logical fixture names are dictionary keys; this module reads and writes no data files.
Source SHA256: 146be15ebf8d7327e6769d15f01af2353d6d4b9978737cf529d245c31bed6336
Operation: gdn_gate_chunk_cumsum
"""
def make_fixtures(seed=20261030):
    import torch
    from pathlib import PurePosixPath
    SEED = seed
    N = 65539
    selected = op = 'gdn_gate_chunk_cumsum'
    fixtures = {}
    folder = PurePosixPath('') / selected
    gen = torch.Generator(device='cpu').manual_seed(SEED)
    g = torch.randn((4, 2048, 8), generator=gen, dtype=torch.float32)
    a = torch.empty(8, dtype=torch.float32).uniform_(1, 16, generator=gen).log()
    bias = torch.randn((8,), generator=gen, dtype=torch.float32)
    gate = -a.double().exp() * torch.nn.functional.softplus(g.double() + bias.double())
    output = torch.cat([v.cumsum(1) for v in gate.split(64, 1)], 1).float()
    path = folder / 'test_gdn_gate_chunk_cumsum_scalar_kernel_cuda.pt'
    fixtures[path.name] = dict(input_data=dict(g=g, A_log=a, dt_bias=bias, chunk_size=64, scale=1.0), gpu_output=dict(o=output))
    return fixtures
