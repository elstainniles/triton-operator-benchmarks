"""Seeded CPU inputs and independent references, ported from the existing FLA test generator.
Logical fixture names are dictionary keys; this module reads and writes no data files.
Source SHA256: 146be15ebf8d7327e6769d15f01af2353d6d4b9978737cf529d245c31bed6336
Operation: rwkv_mix_bwd
"""
def make_fixtures(seed=20261030):
    import torch
    from pathlib import PurePosixPath
    SEED = seed
    N = 65539
    selected = op = 'rwkv_mix_bwd'
    fixtures = {}
    folder = PurePosixPath('') / selected
    gen = torch.Generator(device='cpu').manual_seed(SEED)
    x = torch.randn((2, 1024, 1024), generator=gen, dtype=torch.float32)
    prev = torch.randn((2, 1, 1024), generator=gen, dtype=torch.float32)
    k = torch.randn((1, 1, 1024), generator=gen, dtype=torch.float32)
    grad = torch.randn(x.shape, generator=gen, dtype=torch.float32)
    xx, pp = (x.double().requires_grad_(True), prev.double().requires_grad_(True))
    previous = torch.cat((pp, xx[:, :-1]), 1)
    y = xx + (previous - xx) * k.double()
    dx, dp = torch.autograd.grad(y, (xx, pp), grad.double())
    path = folder / 'test_rwkv_mix_bwd_kenel_cuda.pt'
    fixtures[path.name] = dict(input_data=dict(x=x, x_prev=prev, x_k=k, dk1=grad, B=2, T=1024, n_embd=1024, dtype=torch.float32), gpu_output=dict(dx=dx.float(), dx_prev=dp.float()))
    return fixtures
