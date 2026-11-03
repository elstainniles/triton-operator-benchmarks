"""Seeded CPU inputs and independent references, ported from the existing FLA test generator.
Logical fixture names are dictionary keys; this module reads and writes no data files.
Source SHA256: 146be15ebf8d7327e6769d15f01af2353d6d4b9978737cf529d245c31bed6336
Operation: logsigmoid_fwd_kernel
"""
def make_fixtures(seed=20261030):
    import torch
    from pathlib import PurePosixPath
    SEED = seed
    N = 65539
    selected = op = 'logsigmoid_fwd_kernel'
    fixtures = {}
    folder = PurePosixPath('') / op
    is_log = op.startswith('logsigmoid')
    is_bwd = '_bwd_' in op
    configs = [(1024, torch.float32, 1.0), (2048, torch.bfloat16, 0.5 if is_log else 1.0)]
    if is_log:
        configs.append((2048, torch.bfloat16, 2.0))
    for dim, dtype, temperature in configs:
        gen = torch.Generator(device='cpu').manual_seed(SEED)
        x = torch.randn((4, 2048, dim), generator=gen, dtype=dtype)
        inp = dict(x=x)
        if is_bwd:
            dy = torch.randn(x.shape, generator=gen, dtype=dtype)
            inp['dy'] = dy
            xx = x.float().detach().requires_grad_(True)
            fn = torch.nn.functional.logsigmoid(xx) / temperature if is_log else torch.sigmoid(xx)
            dx, = torch.autograd.grad(fn, xx, dy.float())
            out = dict(dx=dx.to(dtype))
        else:
            out = dict(y=(torch.nn.functional.logsigmoid(x.float()) / temperature if is_log else torch.sigmoid(x.float())).to(dtype))
        name = ('logsigmoid' if is_log else 'sigmoid') + ('_bwd' if is_bwd else '_fwd') + f'_B4_T2048_D{dim}'
        if is_log:
            name += f'_temp{temperature}'
        for suffix, obj in [('_in.pt', inp), ('_out.pt', out)]:
            path = folder / (name + suffix)
            fixtures[path.name] = obj
    return fixtures
