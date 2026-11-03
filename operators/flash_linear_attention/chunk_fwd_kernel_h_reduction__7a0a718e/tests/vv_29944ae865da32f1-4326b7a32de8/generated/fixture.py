"""Seeded CPU inputs and independent references, ported from the existing FLA test generator.
Logical fixture names are dictionary keys; this module reads and writes no data files.
Source SHA256: 146be15ebf8d7327e6769d15f01af2353d6d4b9978737cf529d245c31bed6336
Operation: chunk_fwd_h_reduction
"""
def make_fixtures(seed=20261030):
    import torch
    from pathlib import PurePosixPath
    SEED = seed
    N = 65539
    selected = op = 'chunk_fwd_h_reduction'
    fixtures = {}
    folder = PurePosixPath('') / selected
    gen = torch.Generator(device='cpu').manual_seed(SEED)
    hs = torch.randn((16, 8, 128, 128), generator=gen)
    hr = torch.zeros_like(hs)
    group = hs.double().reshape(4, 4, 8, 128, 128)
    ref = torch.zeros_like(group)
    ref[:, 1:] = group[:, :-1].cumsum(1)
    path = folder / 'test_chunk_h_split_chunk_fwd_kernel_h_reduction_cuda.pt'
    fixtures[path.name] = dict(input_data=dict(hs=hs, hr=hr, g=None, gk=None, gv=None, output_final_state=False, cu_seqlens=None, split_offsets=None, chunk_size=64, split_size=256), gpu_output=dict(ht=None, hr=ref.float().reshape_as(hs)))
    return fixtures
