"""Seeded CPU inputs and independent references, ported from the existing FLA test generator.
Logical fixture names are dictionary keys; this module reads and writes no data files.
Source SHA256: 146be15ebf8d7327e6769d15f01af2353d6d4b9978737cf529d245c31bed6336
Operation: chunk_comba_cumsum_scalar_fwd
"""
def make_fixtures(seed=20261030):
    import torch
    from pathlib import PurePosixPath
    SEED = seed
    N = 65539
    selected = op = 'chunk_comba_cumsum_scalar_fwd'
    fixtures = {}
    folder = PurePosixPath('') / selected
    gen = torch.Generator(device='cpu').manual_seed(SEED)
    dtype = torch.float32 if selected.endswith('fwd') else torch.float16
    x = torch.randn((32, 2048, 8), generator=gen, dtype=dtype)
    inp = dict(B=32, T=2048, H=8, BT=128, NT=16)
    parts = [a.double().cumsum(1) for a in x.split(128, 1)]
    inclusive = torch.cat(parts, 1)
    exclusive = inclusive - x.double()
    inp.update(g=x, g0=torch.zeros_like(x), g1=torch.zeros_like(x))
    output = dict(g0=exclusive.float(), g1=inclusive.float())
    path = folder / ('test_' + selected + '_kernel_cuda.pt')
    fixtures[path.name] = dict(input_data=inp, gpu_output=output)
    return fixtures
