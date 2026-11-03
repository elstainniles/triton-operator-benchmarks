"""Seeded CPU inputs and independent references, ported from the existing FLA test generator.
Logical fixture names are dictionary keys; this module reads and writes no data files.
Source SHA256: 146be15ebf8d7327e6769d15f01af2353d6d4b9978737cf529d245c31bed6336
Operation: gdn_gate_bwd
"""
def make_fixtures(seed=20261030):
    import torch
    from pathlib import PurePosixPath
    SEED = seed
    N = 65539
    selected = op = 'gdn_gate_bwd'
    fixtures = {}
    folder = PurePosixPath('') / selected
    gen = torch.Generator(device='cpu').manual_seed(SEED)
    g = torch.randn((4096, 64), generator=gen, dtype=torch.float32)
    a = torch.randn((64,), generator=gen, dtype=torch.float32)
    bias = torch.randn((64,), generator=gen, dtype=torch.float32)
    inp = dict(g=g, A_log=a, dt_bias=bias, B=4, T=1024, H=64)
    gd, ad, bd = [t.double().requires_grad_(True) for t in (g, a, bias)]
    y = -ad.exp() * torch.nn.functional.softplus(gd + bd)
    dy = torch.randn(g.shape, generator=gen, dtype=torch.float32)
    dg, da, db = torch.autograd.grad(y, (gd, ad, bd), dy.double())
    inp.update(dyg=dy, dg=torch.zeros_like(g), dA=torch.zeros((128, 64)), BT=32, NT=128)
    output = dict(dg=dg.float(), dA=da.float(), dbias=db.float())
    path = folder / ('test_' + selected + '_kernel_cuda.pt')
    fixtures[path.name] = dict(input_data=inp, gpu_output=output)
    return fixtures
