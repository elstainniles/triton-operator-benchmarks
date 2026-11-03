"""Seeded CPU inputs and independent references, ported from the existing FLA test generator.
Logical fixture names are dictionary keys; this module reads and writes no data files.
Source SHA256: 146be15ebf8d7327e6769d15f01af2353d6d4b9978737cf529d245c31bed6336
Operation: fused_recurrent_rwkv6_fwd
"""
def make_fixtures(seed=20261030):
    import torch
    from pathlib import PurePosixPath
    SEED = seed
    N = 65539
    selected = op = 'fused_recurrent_rwkv6_fwd'
    fixtures = {}
    folder = PurePosixPath('') / selected
    gen = torch.Generator(device='cpu').manual_seed(SEED)
    shape = (4, 2048, 4, 256)
    q = torch.randn(shape, generator=gen, dtype=torch.float16)
    k = torch.randn(shape, generator=gen, dtype=torch.float16)
    v = torch.randn(shape, generator=gen, dtype=torch.float16)
    u = torch.randn((4, 256), generator=gen, dtype=torch.float16)
    h0 = torch.randn((4, 4, 256, 256), generator=gen, dtype=torch.float16)
    w = torch.nn.functional.logsigmoid(torch.randn(shape, generator=gen, dtype=torch.float16))
    scale = 256 ** (-0.5)
    h = h0.double()
    out = torch.empty(shape, dtype=torch.float32)
    uu = u.double()
    for t in range(2048):
        qt = q[:, t].double() * scale
        kt = k[:, t].double()
        vt = v[:, t].double()
        out[:, t] = (torch.einsum('bhk,bhkv->bhv', qt, h) + (qt * uu * kt).sum(-1)[:, :, None] * vt).float()
        h = h * w[:, t].double().exp()[:, :, :, None] + kt[:, :, :, None] * vt[:, :, None, :]
    inp = dict(q=q, k=k, v=v, w=w, u=u, h0=h0, scale=scale, output_final_state=True, reverse=False, B=4, T=2048, H=4, K=256, V=256)
    path = folder / 'test_fused_recurrent_rwkv6_fwd_kernel_cuda.pt'
    fixtures[path.name] = dict(input_data=inp, gpu_output=dict(o=out, ht=h.float()))
    return fixtures
