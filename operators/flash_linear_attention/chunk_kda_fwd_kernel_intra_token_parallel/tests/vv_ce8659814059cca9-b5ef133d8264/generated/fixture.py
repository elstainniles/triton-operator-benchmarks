"""Seeded CPU inputs and independent references, ported from the existing FLA test generator.
Logical fixture names are dictionary keys; this module reads and writes no data files.
Source SHA256: 146be15ebf8d7327e6769d15f01af2353d6d4b9978737cf529d245c31bed6336
Operation: chunk_kda_intra_token
"""
def make_fixtures(seed=20261030):
    import torch
    from pathlib import PurePosixPath
    SEED = seed
    N = 65539
    selected = op = 'chunk_kda_intra_token'
    fixtures = {}
    folder = PurePosixPath('') / selected
    gen = torch.Generator(device='cpu').manual_seed(SEED)
    shape = (2, 2048, 8, 64)
    q = torch.randn(shape, generator=gen, dtype=torch.float16) * 0.1
    k = torch.randn(shape, generator=gen, dtype=torch.float16) * 0.1
    g = torch.randn(shape, generator=gen, dtype=torch.float16) * 0.1
    beta = torch.randn((2, 2048, 8), generator=gen, dtype=torch.float16).sigmoid()
    aq = torch.randn((2, 2048, 8, 64), generator=gen, dtype=torch.float16) * 0.1
    ak = torch.randn((2, 2048, 8, 16), generator=gen, dtype=torch.float16) * 0.1
    oq = aq.clone()
    ok = ak.clone()
    scale = 64 ** (-0.5)
    for t in range(0, 2048, 16):
        qq = q[:, t:t + 16].double().permute(0, 2, 1, 3)
        kk = k[:, t:t + 16].double().permute(0, 2, 1, 3)
        gg = g[:, t:t + 16].double().permute(0, 2, 1, 3)
        bb = beta[:, t:t + 16].double().permute(0, 2, 1)
        vq = torch.matmul(qq * torch.exp2(gg), (kk * torch.exp2(-gg)).transpose(-1, -2)) * scale
        vk = torch.matmul(kk * torch.exp2(gg) * bb[..., None], (kk * torch.exp2(-gg)).transpose(-1, -2))
        for row in range(16):
            oq[:, t + row, :, t % 64:t % 64 + row + 1] = vq[:, :, row, :row + 1].half()
            ok[:, t + row, :, :row + 1] = vk[:, :, row, :row + 1].half()
            ok[:, t + row, :, row] = 0
    inp = dict(q=q, k=k, gk=g, beta=beta, Aqk=aq, Akk=ak, scale=scale, chunk_size=64, sub_chunk_size=16)
    path = folder / 'test_chunk_kda_fwd_kernel_intra_token_parallel_cuda.pt'
    fixtures[path.name] = dict(input_data=inp, gpu_output=dict(Aqk=oq, Akk=ok))
    return fixtures
