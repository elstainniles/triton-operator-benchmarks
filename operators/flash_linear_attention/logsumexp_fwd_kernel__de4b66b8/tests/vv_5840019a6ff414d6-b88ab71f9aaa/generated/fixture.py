"""Seeded CPU inputs and independent references, ported from the existing FLA test generator.
Logical fixture names are dictionary keys; this module reads and writes no data files.
Source SHA256: 146be15ebf8d7327e6769d15f01af2353d6d4b9978737cf529d245c31bed6336
Operation: logsumexp_fwd
"""
def make_fixtures(seed=20261030):
    import torch
    from pathlib import PurePosixPath
    SEED = seed
    N = 65539
    selected = op = 'logsumexp_fwd'
    fixtures = {}
    folder = PurePosixPath('') / selected
    gen = torch.Generator(device='cpu').manual_seed(SEED)
    x = torch.randn((2, 1024, 128), generator=gen, dtype=torch.bfloat16)
    z = torch.logsumexp(x.double() * 0.5, dim=-1).float()
    path = folder / 'test_logsumexp_fwd_kernel_cuda.pt'
    fixtures[path.name] = dict(input_data=dict(x=x, scale=0.5, B=2, T=1024, D=128), gpu_output=dict(z=z))
    return fixtures
