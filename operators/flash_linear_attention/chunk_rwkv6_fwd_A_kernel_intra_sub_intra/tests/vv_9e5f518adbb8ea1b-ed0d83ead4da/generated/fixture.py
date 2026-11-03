"""Seeded CPU inputs and independent references, ported from the existing FLA test generator.
Logical fixture names are dictionary keys; this module reads and writes no data files.
Source SHA256: 146be15ebf8d7327e6769d15f01af2353d6d4b9978737cf529d245c31bed6336
Operation: chunk_rwkv6_intra
"""
def make_fixtures(seed=20261030):
    import torch
    from pathlib import PurePosixPath
    SEED = seed
    N = 65539
    selected = op = 'chunk_rwkv6_intra'
    fixtures = {}
    folder = PurePosixPath('') / selected
    gen = torch.Generator(device='cpu').manual_seed(SEED)
    shape = (4, 1000, 4, 64)
    q = torch.randn(shape, generator=gen, dtype=torch.float16)
    k = torch.randn(shape, generator=gen, dtype=torch.float16)
    u = torch.randn((4, 64), generator=gen, dtype=torch.float16)
    gi = torch.randn(shape, generator=gen) * 0.1
    ge = torch.randn(shape, generator=gen) * 0.1
    initial = torch.randn((4, 1000, 4, 64), generator=gen) * 0.1
    out = initial.clone()
    scale = 64 ** (-0.5)
    for t in range(0, 1000, 16):
        n = min(16, 1000 - t)
        slot = t % 64
        qq = q[:, t:t + n].double().permute(0, 2, 1, 3)
        kk = k[:, t:t + n].double().permute(0, 2, 1, 3)
        eg = ge[:, t:t + n].double().permute(0, 2, 1, 3)
        ig = gi[:, t:t + n].double().permute(0, 2, 1, 3)
        val = torch.tril(torch.matmul(qq * torch.exp2(eg), (kk * torch.exp2(-ig)).transpose(-1, -2)) * scale, diagonal=-1)
        diag = (qq * kk * u.double()[None, :, None, :]).sum(-1) * scale
        val = val + torch.diag_embed(diag)
        out[:, t:t + n, :, slot:slot + n] = val.permute(0, 2, 1, 3).float()
    inp = dict(q=q, k=k, gi=gi, ge=ge, u=u, A=initial, scale=scale, T=1000, H=4, K=64, BT=64, BC=16, BK=64)
    for name, obj in [('inputs.pt', inp), ('outputs.pt', dict(A_out=out)), ('outputs_fp32.pt', dict(A_out_fp32=out))]:
        path = folder / name
        fixtures[path.name] = obj
    return fixtures
