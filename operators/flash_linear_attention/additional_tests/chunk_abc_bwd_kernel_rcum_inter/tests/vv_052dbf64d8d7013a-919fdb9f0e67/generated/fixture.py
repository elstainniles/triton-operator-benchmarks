"""Seeded CPU inputs and independent references, ported from the existing FLA test generator.
Logical fixture names are dictionary keys; this module reads and writes no data files.
Source SHA256: 146be15ebf8d7327e6769d15f01af2353d6d4b9978737cf529d245c31bed6336
Operation: chunk_abc_bwd_rcum_inter
"""
def make_fixtures(seed=20261030):
    import torch
    from pathlib import PurePosixPath
    SEED = seed
    N = 65539
    selected = op = 'chunk_abc_bwd_rcum_inter'
    fixtures = {}
    folder = PurePosixPath('') / selected
    gen = torch.Generator(device='cpu').manual_seed(SEED)
    s = torch.rand((2, 4, 512, 16), generator=gen, dtype=torch.bfloat16)
    z = s.float().logcumsumexp(2).contiguous()
    ss = torch.rand(s.shape, generator=gen)
    doo = torch.zeros_like(s)
    weighted = ss.double() * (-z.double()).exp()
    parts = weighted.reshape(2, 4, 8, 64, 16).sum(3)
    future = torch.zeros_like(parts)
    future[:, :, :-1] = parts[:, :, 1:].flip(2).cumsum(2).flip(2)
    output = (s.double().exp() * future[:, :, :, None, :].expand(2, 4, 8, 64, 16).reshape_as(s)).bfloat16()
    path = folder / 'test_chunk_abc_bwd_kernel_rcum_inter.pt'
    fixtures[path.name] = dict(input_data=dict(s=s, z=z, ss=ss, doo=doo, T=512, S=16, BT=64, BS=16, NT=8, num_warps=4, num_stages=1), gpu_output=dict(doo=output), grid=(1, 8))
    return fixtures
