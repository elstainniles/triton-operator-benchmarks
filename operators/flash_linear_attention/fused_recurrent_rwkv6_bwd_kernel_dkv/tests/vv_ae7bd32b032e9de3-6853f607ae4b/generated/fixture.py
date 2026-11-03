"""Seeded CPU inputs and independent references, ported from the existing FLA test generator.
Logical fixture names are dictionary keys; this module reads and writes no data files.
Source SHA256: 146be15ebf8d7327e6769d15f01af2353d6d4b9978737cf529d245c31bed6336
Operation: fused_recurrent_rwkv6_bwd_dkv
"""
def make_fixtures(seed=20261030):
    import torch
    from pathlib import PurePosixPath
    SEED = seed
    N = 65539
    selected = op = 'fused_recurrent_rwkv6_bwd_dkv'
    fixtures = {}
    folder = PurePosixPath('') / selected
    gen = torch.Generator(device='cpu').manual_seed(SEED)
    shape = (4, 2048, 4, 256)
    q = torch.randn(shape, generator=gen, dtype=torch.float16)
    k = torch.randn(shape, generator=gen, dtype=torch.float16)
    v = torch.randn(shape, generator=gen, dtype=torch.float16)
    w = torch.nn.functional.logsigmoid(torch.randn(shape, generator=gen, dtype=torch.float16))
    u = torch.randn((4, 256), generator=gen, dtype=torch.float16)
    do = torch.randn(shape, generator=gen, dtype=torch.float16)
    h0 = torch.randn((4, 4, 256, 256), generator=gen, dtype=torch.float16)
    scale = 256 ** (-0.5)
    dk = torch.empty(shape, dtype=torch.float32)
    dk1 = torch.empty_like(dk)
    dv = torch.empty_like(dk)
    state = torch.zeros((4, 4, 256, 256), dtype=torch.float64)
    uu = u.double()
    for t in range(2047, -1, -1):
        qt = q[:, t].double() * scale
        kt = k[:, t].double()
        vt = v[:, t].double()
        gt = do[:, t].double()
        wt = w[:, t].double()
        future = torch.matmul(state, vt.unsqueeze(-1)).squeeze(-1)
        dk1[:, t] = future.float()
        dk[:, t] = (future + qt * uu * (gt * vt).sum(-1, keepdim=True)).float()
        dv[:, t] = (torch.matmul(state.transpose(-1, -2), kt.unsqueeze(-1)).squeeze(-1) + gt * (qt * uu * kt).sum(-1, keepdim=True)).float()
        state = state * torch.exp(wt).unsqueeze(-1) + qt.unsqueeze(-1) * gt.unsqueeze(-2)
    inp = dict(q=q, k=k, v=v, w=w, u=u, do=do, h0=h0, scale=scale, reverse=False, B=4, T=2048, H=4, K=256, V=256)
    path = folder / 'test_fused_recurrent_rwkv6_bwd_kernel_dkv_cuda.pt'
    fixtures[path.name] = dict(input_data=inp, gpu_output=dict(dk=dk, dk1=dk1, dv=dv, dh0=state.half()))
    return fixtures
