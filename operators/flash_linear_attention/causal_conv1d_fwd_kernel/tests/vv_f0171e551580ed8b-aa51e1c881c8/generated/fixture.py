"""Seeded CPU inputs and independent references, ported from the existing FLA test generator.
Logical fixture names are dictionary keys; this module reads and writes no data files.
Source SHA256: 146be15ebf8d7327e6769d15f01af2353d6d4b9978737cf529d245c31bed6336
Operation: causal_conv1d_fwd
"""
def make_fixtures(seed=20261030):
    import torch
    from pathlib import PurePosixPath
    SEED = seed
    N = 65539
    selected = op = 'causal_conv1d_fwd'
    fixtures = {}
    folder = PurePosixPath('') / selected
    gen = torch.Generator(device='cpu').manual_seed(SEED)
    x = torch.randn((2, 1024, 128), generator=gen)
    weight = torch.randn((128, 3), generator=gen)
    bias = torch.randn((128,), generator=gen)
    xc = x.double().transpose(1, 2)
    padded = torch.nn.functional.pad(xc, (2, 0))
    out = torch.nn.functional.conv1d(padded, weight.double().unsqueeze(1), bias.double(), groups=128).transpose(1, 2).float()
    path = folder / 'test_causal_conv1d_fwd_kernel_cuda.pt'
    fixtures[path.name] = dict(input_data=dict(x=x, weight=weight, bias=bias, B=2, T=1024, D=128, W=3), gpu_output=dict(y=out))
    return fixtures
