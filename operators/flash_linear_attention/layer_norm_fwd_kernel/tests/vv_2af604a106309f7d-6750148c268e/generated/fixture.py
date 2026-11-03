"""Seeded CPU inputs and independent references, ported from the existing FLA test generator.
Logical fixture names are dictionary keys; this module reads and writes no data files.
Source SHA256: 146be15ebf8d7327e6769d15f01af2353d6d4b9978737cf529d245c31bed6336
Operation: layer_norm_fwd
"""
def make_fixtures(seed=20261030):
    import torch
    from pathlib import PurePosixPath
    SEED = seed
    N = 65539
    selected = op = 'layer_norm_fwd'
    fixtures = {}
    folder = PurePosixPath('') / selected
    gen = torch.Generator(device='cpu').manual_seed(SEED)
    dim = 1024 if selected == 'layer_norm_fwd1' else 128
    x = torch.randn((2048, dim), generator=gen, dtype=torch.float32)
    weight = torch.randn((dim,), generator=gen, dtype=torch.float32)
    bias = torch.randn((dim,), generator=gen, dtype=torch.float32)
    xx, ww, bb = [t.double().requires_grad_(True) for t in (x, weight, bias)]
    mean = xx.mean(-1)
    rstd = (xx.var(-1, unbiased=False) + 1e-05).rsqrt()
    yy = (xx - mean[:, None]) * rstd[:, None] * ww + bb
    inp = dict(x=x, weight=weight, bias=bias, eps=1e-05, T=2048, D=dim)
    output = dict(y=yy.float(), mean=mean.float(), rstd=rstd.float())
    name = {'layer_norm_fwd': 'test_layer_norm_fwd_kernel_cuda.pt', 'layer_norm_fwd1': 'test_layer_norm_fwd_kernel1_cuda.pt', 'layer_norm_bwd': 'test_layer_norm_bwd_kernel_cuda.pt'}[selected]
    path = folder / name
    fixtures[path.name] = dict(input_data=inp, gpu_output=output)
    return fixtures
