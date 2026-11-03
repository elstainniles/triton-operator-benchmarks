"""Seeded CPU inputs and independent references, ported from the existing FLA test generator.
Logical fixture names are dictionary keys; this module reads and writes no data files.
Source SHA256: 146be15ebf8d7327e6769d15f01af2353d6d4b9978737cf529d245c31bed6336
Operation: fused_recurrent_comba_fwd
"""
def make_fixtures(seed=20261030):
    import torch
    from pathlib import PurePosixPath
    SEED = seed
    N = 65539
    selected = op = 'fused_recurrent_comba_fwd'
    fixtures = {}
    folder = PurePosixPath('') / selected
    gen = torch.Generator(device='cpu').manual_seed(SEED)

    def norm():
        return torch.nn.functional.normalize(torch.randn((2, 1024, 8, 128), generator=gen), p=2, dim=-1)
    q = norm()
    k = norm()
    v = torch.randn(q.shape, generator=gen)
    p = norm()
    beta = torch.rand((2, 1024, 8), generator=gen).sigmoid()
    g = torch.nn.functional.logsigmoid(torch.rand((2, 1024, 8), generator=gen))
    h0 = torch.randn((2, 8, 128, 128), generator=gen)
    state = h0.double()
    out = torch.empty_like(q)
    for t in range(1024):
        qt = q[:, t].double()
        kt = k[:, t].double()
        pt = p[:, t].double()
        vt = v[:, t].double()
        bt = beta[:, t].double()
        gt = g[:, t].double()
        corrected = (vt - torch.einsum('bhk,bhkv->bhv', pt, state)) * bt[:, :, None]
        state = state * gt.exp()[:, :, None, None] + kt[:, :, :, None] * corrected[:, :, None, :]
        out[:, t] = torch.einsum('bhk,bhkv->bhv', qt, state).float()
    inp = dict(q=q, k=k, v=v, p=p, beta=beta, g=g, h0=h0, o=torch.zeros((1, 2, 1024, 8, 128)), final_state=torch.zeros_like(h0), B=2, T=1024, H=8, D=128, HV=8, K=128, V=128, BK=128, BV=8, NK=1, NV=16, scale=1.0)
    path = folder / 'test_fused_recurrent_comba_fwd_kernel_cuda.pt'
    fixtures[path.name] = dict(input_data=inp, gpu_output=dict(o=out, final_state=state.float()))
    return fixtures
