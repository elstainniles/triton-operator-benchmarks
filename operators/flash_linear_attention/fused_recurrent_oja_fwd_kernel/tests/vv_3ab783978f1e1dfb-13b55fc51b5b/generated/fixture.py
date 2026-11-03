"""Seeded CPU inputs and independent references, ported from the existing FLA test generator.
Logical fixture names are dictionary keys; this module reads and writes no data files.
Source SHA256: 146be15ebf8d7327e6769d15f01af2353d6d4b9978737cf529d245c31bed6336
Operation: fused_recurrent_oja_fwd
"""
def make_fixtures(seed=20261030):
    import torch
    from pathlib import PurePosixPath
    SEED = seed
    N = 65539
    selected = op = 'fused_recurrent_oja_fwd'
    fixtures = {}
    folder = PurePosixPath('') / selected
    gen = torch.Generator(device='cpu').manual_seed(SEED)
    shape = (2, 8, 4, 64)
    q = torch.randn(shape, generator=gen, dtype=torch.bfloat16)
    k = torch.randn(shape, generator=gen, dtype=torch.bfloat16)
    v = torch.nn.functional.normalize(torch.randn(shape, generator=gen), p=2, dim=-1).bfloat16()
    gv = torch.nn.functional.logsigmoid(torch.rand(shape, generator=gen))
    beta = torch.rand((2, 8, 4), generator=gen, dtype=torch.bfloat16).sigmoid()
    h0 = torch.randn((2, 4, 64, 64), generator=gen)
    scale = 64 ** (-0.5)
    state = h0.double()
    out = torch.empty_like(v)
    for t in range(8):
        qt = q[:, t].double() * scale
        kt = k[:, t].double()
        vt = v[:, t].double()
        state = state * gv[:, t].double().exp()[:, :, None, :]
        adjusted = (kt - torch.einsum('bhkv,bhv->bhk', state, vt)) * beta[:, t].double()[:, :, None]
        state = state + adjusted[:, :, :, None] * vt[:, :, None, :]
        out[:, t] = torch.einsum('bhk,bhkv->bhv', qt, state).bfloat16()
    inp = dict(q=q, k=k, v=v, gv=gv, beta=beta, o=torch.zeros_like(v), h0=h0, ht=torch.zeros_like(h0), cu_seqlens=None, scale=scale, T=8, B=2, H=4, HV=4, K=64, V=64, BK=64, BV=64, IS_BETA_HEADWISE=True, USE_Q_L2NORM=False, USE_K_L2NORM=False, num_warps=1, num_stages=3)
    path = folder / 'test_fused_recurrent_oja_fwd_kernel.pt'
    fixtures[path.name] = dict(input_data=inp, gpu_output=dict(o=out, ht=state.float()), grid=(1, 8))
    return fixtures
