"""Seeded CPU inputs and independent references, ported from the existing FLA test generator.
Logical fixture names are dictionary keys; this module reads and writes no data files.
Source SHA256: 146be15ebf8d7327e6769d15f01af2353d6d4b9978737cf529d245c31bed6336
Operation: chunk_scaled_dot_kkt_intra
"""
def make_fixtures(seed=20261030):
    import torch
    from pathlib import PurePosixPath
    SEED = seed
    N = 65539
    selected = op = 'chunk_scaled_dot_kkt_intra'
    fixtures = {}
    folder = PurePosixPath('') / selected
    gen = torch.Generator(device='cpu').manual_seed(SEED)
    shape = (2, 2048, 16, 128)
    k = torch.randn(shape, generator=gen) * 0.1
    g = torch.randn(shape, generator=gen) * 0.1
    beta = torch.rand((2, 2048, 16), generator=gen).sigmoid()
    out = torch.zeros((2, 2048, 16, 64), dtype=torch.float32)
    for t in range(0, 2048, 16):
        kk = k[:, t:t + 16].double().permute(0, 2, 1, 3)
        gg = g[:, t:t + 16].double().permute(0, 2, 1, 3)
        bb = beta[:, t:t + 16].double().permute(0, 2, 1)
        v = torch.tril(torch.matmul(kk * torch.exp(gg) * bb[..., None], (kk * torch.exp(-gg)).transpose(-1, -2)), diagonal=-1)
        slot = t % 64
        out[:, t:t + 16, :, slot:slot + 16] = v.permute(0, 2, 1, 3).float()
    inp = dict(k=k, g=g, beta=beta, A=torch.zeros_like(out), B=2, T=2048, H=16, K=128, BT=64, BC=16, BK=128, NT=32, NC=4)
    path = folder / 'test_chunk_scaled_dot_kkt_fwd_kernel_intra_sub_intra_cuda.pt'
    fixtures[path.name] = dict(input_data=inp, gpu_output=dict(A=out))
    return fixtures
