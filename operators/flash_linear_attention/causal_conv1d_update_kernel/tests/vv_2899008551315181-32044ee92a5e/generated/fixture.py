"""Seeded CPU inputs and independent references, ported from the existing FLA test generator.
Logical fixture names are dictionary keys; this module reads and writes no data files.
Source SHA256: 146be15ebf8d7327e6769d15f01af2353d6d4b9978737cf529d245c31bed6336
Operation: causal_conv1d_update
"""
def make_fixtures(seed=20261030):
    import torch
    from pathlib import PurePosixPath
    SEED = seed
    N = 65539
    selected = op = 'causal_conv1d_update'
    fixtures = {}
    folder = PurePosixPath('') / selected
    gen = torch.Generator(device='cpu').manual_seed(SEED)
    x = torch.randn((2, 1, 128), generator=gen)
    cache = torch.randn((2, 128, 3), generator=gen)
    weight = torch.randn((128, 3), generator=gen)
    bias = torch.randn((128,), generator=gen)
    shifted = torch.cat((cache[:, :, 1:], x[:, 0, :, None]), dim=2)
    out = ((shifted.double() * weight.double()).sum(dim=2) + bias.double()).float().unsqueeze(1)
    path = folder / 'test_causal_conv1d_update_kernel_cuda.pt'
    fixtures[path.name] = dict(input_data=dict(x=x, cache=cache, weight=weight, bias=bias, B=2, D=128, W=3), gpu_output=dict(y=out, cache=shifted))
    return fixtures
