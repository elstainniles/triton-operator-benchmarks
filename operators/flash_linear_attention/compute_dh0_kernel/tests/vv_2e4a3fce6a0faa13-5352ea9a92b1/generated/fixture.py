"""Seeded CPU inputs and independent references, ported from the existing FLA test generator.
Logical fixture names are dictionary keys; this module reads and writes no data files.
Source SHA256: 146be15ebf8d7327e6769d15f01af2353d6d4b9978737cf529d245c31bed6336
Operation: compute_dh0
"""
def make_fixtures(seed=20261030):
    import torch
    from pathlib import PurePosixPath
    SEED = seed
    N = 65539
    selected = op = 'compute_dh0'
    fixtures = {}
    folder = PurePosixPath('') / selected
    gen = torch.Generator(device='cpu').manual_seed(SEED)
    dy = torch.randn((4, 1024, 4096), generator=gen)
    y = torch.randn((4, 1024, 4096), generator=gen)
    weight = torch.randn((4096, 4), generator=gen)
    initial = torch.randn((4, 4096, 4), generator=gen)
    h = initial.double().requires_grad_(True)
    extended = torch.cat((h[:, :, 1:], torch.zeros((4, 4096, 3), dtype=torch.float64)), dim=2)
    z = torch.nn.functional.conv1d(extended, weight.double().unsqueeze(1), groups=4096).transpose(1, 2)
    yy = y[:, :3].double()
    sig = yy.sigmoid()
    g = dy[:, :3].double() * sig * (1 + yy * (1 - sig))
    dh, = torch.autograd.grad(z, h, g)
    for name, obj in [('compute_dh0_kernel_in.pt', dict(dy=dy, y=y, weight=weight, initial_state=initial, activation='swish', seq_idx=None)), ('compute_dh0_kernel_out.pt', dict(dh0=dh.float()))]:
        path = folder / name
        fixtures[path.name] = obj
    return fixtures
