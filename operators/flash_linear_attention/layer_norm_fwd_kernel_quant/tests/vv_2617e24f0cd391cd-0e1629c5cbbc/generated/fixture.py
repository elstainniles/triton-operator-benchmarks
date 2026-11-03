"""Seeded CPU inputs and independent references, ported from the existing FLA test generator.
Logical fixture names are dictionary keys; this module reads and writes no data files.
Source SHA256: 146be15ebf8d7327e6769d15f01af2353d6d4b9978737cf529d245c31bed6336
Operation: layer_norm_fwd_quant
"""
def make_fixtures(seed=20261030):
    import torch
    from pathlib import PurePosixPath
    SEED = seed
    N = 65539
    selected = op = 'layer_norm_fwd_quant'
    fixtures = {}
    folder = PurePosixPath('') / selected
    gen = torch.Generator(device='cpu').manual_seed(SEED)
    weight = torch.randn((128,), generator=gen)
    bias = torch.randn((128,), generator=gen)
    xs = []
    remaining = 2048
    eps = 1e-05
    while remaining:
        sample = torch.randn((remaining * 2, 128), generator=gen)
        x = sample.double()
        mu = x.mean(-1, keepdim=True)
        rstd = (x.var(-1, unbiased=False, keepdim=True) + eps).rsqrt()
        y = (x - mu) * rstd * weight.double() + bias.double()
        scale = 127 / y.abs().amax(-1, keepdim=True).clamp_min(1e-05)
        z = y * scale
        good = (torch.abs(z - torch.floor(z) - 0.5) > 0.002).all(-1)
        chosen = sample[good][:remaining]
        xs.append(chosen)
        remaining -= len(chosen)
    x = torch.cat(xs)
    xd = x.double()
    mean = xd.mean(-1)
    rstd = (xd.var(-1, unbiased=False) + eps).rsqrt()
    y = (xd - mean[:, None]) * rstd[:, None] * weight.double() + bias.double()
    scale = 127 / y.abs().amax(-1, keepdim=True).clamp_min(1e-05)
    z = y * scale
    quant = (torch.sign(z) * torch.floor(z.abs() + 0.5)).clamp(-128, 127) / scale
    inp = dict(x=x, weight=weight, bias=bias, eps=eps, is_rms_norm=False, M=2048, N=128)
    path = folder / 'test_layer_norm_fwd_kernel_quant_cuda.pt'
    fixtures[path.name] = dict(input_data=inp, gpu_output=dict(y=quant.float(), mean=mean.float(), rstd=rstd.float()))
    return fixtures
