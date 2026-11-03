"""Seeded CPU inputs and independent references, ported from the existing FLA test generator.
Logical fixture names are dictionary keys; this module reads and writes no data files.
Source SHA256: 146be15ebf8d7327e6769d15f01af2353d6d4b9978737cf529d245c31bed6336
Operation: packunpack_sequence
"""
def make_fixtures(seed=20261030):
    import torch
    from pathlib import PurePosixPath
    SEED = seed
    N = 65539
    selected = op = 'packunpack_sequence'
    fixtures = {}
    folder = PurePosixPath('') / selected
    gen = torch.Generator(device='cpu').manual_seed(SEED)
    lengths = torch.randint(256, 513, (4,), generator=gen, dtype=torch.int64)
    cu = torch.cat((torch.zeros(1, dtype=torch.int64), lengths.cumsum(0)))
    total = int(cu[-1])
    x = torch.randn((4, 512, 256), generator=gen, dtype=torch.float32) * 0.1
    out = torch.cat([x[b, 512 - int(lengths[b]):] for b in range(4)], dim=0)
    path = folder / 'test_packunpack_sequence_kernel_cuda.pt'
    fixtures[path.name] = dict(input_data=dict(x=x, cu_seqlens=cu, y=torch.zeros((total, 256)), B=4, S=512, D=256, BD=256, ND=1, total_len=total, padding_side='left'), gpu_output=dict(y=out))
    return fixtures
