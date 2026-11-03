"""Seeded CPU inputs and independent references, ported from the existing FLA test generator.
Logical fixture names are dictionary keys; this module reads and writes no data files.
Source SHA256: 146be15ebf8d7327e6769d15f01af2353d6d4b9978737cf529d245c31bed6336
Operation: chunk_rwkv6_intra_split
"""
def make_fixtures(seed=20261030):
    import torch
    from pathlib import PurePosixPath
    SEED = seed
    N = 65539
    selected = op = 'chunk_rwkv6_intra_split'
    fixtures = {}
    folder = PurePosixPath('') / selected
    gen = torch.Generator(device='cpu').manual_seed(SEED)
    shape = (4, 2048, 4, 512)
    q = torch.randn(shape, generator=gen)
    k = torch.randn(shape, generator=gen)
    gi = torch.randn(shape, generator=gen)
    ge = torch.randn(shape, generator=gen)
    u = torch.randn((4, 512), generator=gen)
    scale = 512 ** (-0.5)
    out = torch.empty((4, 4, 2048, 4, 16), dtype=torch.float32)
    for part in range(4):
        sel = slice(part * 128, (part + 1) * 128)
        qq = q[..., sel].double().reshape(4, 128, 16, 4, 128).permute(0, 1, 3, 2, 4)
        kk = k[..., sel].double().reshape(4, 128, 16, 4, 128).permute(0, 1, 3, 2, 4)
        eg = ge[..., sel].double().reshape(4, 128, 16, 4, 128).permute(0, 1, 3, 2, 4)
        ig = gi[..., sel].double().reshape(4, 128, 16, 4, 128).permute(0, 1, 3, 2, 4)
        val = torch.tril(torch.matmul(qq * torch.exp2(eg), (kk * torch.exp2(-ig)).transpose(-1, -2)) * scale, diagonal=-1)
        diag = (qq * kk * u[:, sel].double()[None, None, :, None, :]).sum(-1) * scale
        val = val + torch.diag_embed(diag)
        out[part] = val.permute(0, 1, 3, 2, 4).reshape(4, 2048, 4, 16).float()
    inp = dict(q=q, k=k, gi=gi, ge=ge, u=u, scale=scale, chunk_size=64, B=4, T=2048, H=4, K=512)
    path = folder / 'test_chunk_rwkv6_fwd_A_kernel_intra_sub_intra_split_cuda.pt'
    fixtures[path.name] = dict(input_data=inp, gpu_output=dict(A_intra=out))
    return fixtures
