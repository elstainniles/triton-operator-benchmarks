"""Seeded CPU inputs and independent references, ported from the existing FLA test generator.
Logical fixture names are dictionary keys; this module reads and writes no data files.
Source SHA256: 146be15ebf8d7327e6769d15f01af2353d6d4b9978737cf529d245c31bed6336
Operation: causal_conv1d_bwd
"""
def make_fixtures(seed=20261030):
    import torch
    from pathlib import PurePosixPath
    SEED = seed
    N = 65539
    selected = op = 'causal_conv1d_bwd'
    fixtures = {}
    folder = PurePosixPath('') / selected
    gen = torch.Generator(device='cpu').manual_seed(SEED)
    x = torch.randn((2, 1024, 64), generator=gen)
    weight = torch.randn((64, 3), generator=gen)
    bias = torch.randn((64,), generator=gen)
    dy = torch.randn(x.shape, generator=gen)
    xx = x.double().requires_grad_(True)
    ww = weight.double().requires_grad_(True)
    bb = bias.double().requires_grad_(True)
    conv = torch.nn.functional.conv1d(torch.nn.functional.pad(xx.transpose(1, 2), (2, 0)), ww.unsqueeze(1), bb, groups=64).transpose(1, 2)
    dx, dw, db = torch.autograd.grad(conv, (xx, ww, bb), dy.double())
    path = folder / 'test_causal_conv1d_bwd_kernel_cuda.pt'
    fixtures[path.name] = dict(input_data=dict(x=x, weight=weight, bias=bias, dy=dy, B=2, T=1024, D=64, W=3), gpu_output=dict(dx=dx.float(), dw=dw.float(), db=db.float()))
    return fixtures
