"""Seeded CPU inputs and independent references, ported from the existing FLA test generator.
Logical fixture names are dictionary keys; this module reads and writes no data files.
Source SHA256: 146be15ebf8d7327e6769d15f01af2353d6d4b9978737cf529d245c31bed6336
Operation: chunk_bwd_dh_reduction
"""
def make_fixtures(seed=20261030):
    import torch
    from pathlib import PurePosixPath
    SEED = seed
    N = 65539
    selected = op = 'chunk_bwd_dh_reduction'
    fixtures = {}
    folder = PurePosixPath('') / selected
    gen = torch.Generator(device='cpu').manual_seed(SEED)
    q = torch.randn((2, 2048, 8, 128), generator=gen)
    k = torch.randn(q.shape, generator=gen)
    v = torch.randn(q.shape, generator=gen)
    do = torch.randn(q.shape, generator=gen)
    scale = 128 ** (-0.5)
    qq = q.double().reshape(2, 8, 256, 8, 128).permute(0, 1, 3, 4, 2)
    dd = do.double().reshape(2, 8, 256, 8, 128).permute(0, 1, 3, 2, 4)
    split = torch.matmul(qq * scale, dd)
    result = torch.zeros_like(split)
    result[:, :-1] = split[:, 1:].flip(1).cumsum(1).flip(1)
    path = folder / 'test_chunk_h_split_chunk_bwd_kernel_dh_reduction_cuda.pt'
    fixtures[path.name] = dict(input_data=dict(q=q, k=k, v=v, do=do, g=None, gk=None, gv=None, h0=None, dht=None, scale=scale, chunk_size=64, split_size=256), gpu_output=dict(dhr=result.float().reshape(16, 8, 128, 128)))
    return fixtures
