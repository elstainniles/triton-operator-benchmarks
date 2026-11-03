"""Seeded CPU inputs and independent references, ported from the existing FLA test generator.
Logical fixture names are dictionary keys; this module reads and writes no data files.
Source SHA256: 146be15ebf8d7327e6769d15f01af2353d6d4b9978737cf529d245c31bed6336
Operation: fused_recurrent_kda_fwd
"""
def make_fixtures(seed=20261030):
    import torch
    from pathlib import PurePosixPath
    SEED = seed
    N = 65539
    selected = op = 'fused_recurrent_kda_fwd'
    fixtures = {}
    folder = PurePosixPath('') / selected
    gen = torch.Generator(device='cpu').manual_seed(SEED)
    shape = (2, 512, 4, 64)
    q = torch.nn.functional.normalize(torch.rand(shape, generator=gen), dim=-1)
    k = torch.nn.functional.normalize(torch.rand(shape, generator=gen), dim=-1)
    v = torch.rand(shape, generator=gen)
    g = torch.nn.functional.logsigmoid(torch.randn(shape, generator=gen))
    beta = torch.randn((2, 512, 4), generator=gen).sigmoid()
    h0 = torch.randn((2, 4, 64, 64), generator=gen)
    h = h0.double()
    out = torch.empty_like(v)
    for t in range(512):
        qt = q[:, t].double()
        kt = k[:, t].double()
        vt = v[:, t].double()
        h = h * g[:, t].double().exp()[:, :, :, None]
        corrected = (vt - torch.einsum('bhk,bhkv->bhv', kt, h)) * beta[:, t].double()[:, :, None]
        h = h + kt[:, :, :, None] * corrected[:, :, None, :]
        out[:, t] = torch.einsum('bhk,bhkv->bhv', qt, h).float()
    inp = dict(q=q, k=k, v=v, g=g, beta=beta, h0=h0, out=torch.zeros_like(v), final_state=torch.zeros_like(h0), B=2, T=512, H=4, HV=4, K=64, V=64, BK=64, BV=32, scale=1.0, stride_init_state_token=h0.stride(0), stride_final_state_token=h0.stride(0))
    path = folder / 'test_fused_recurrent_kda_fwd_kernel_cuda.pt'
    fixtures[path.name] = dict(input_data=inp, gpu_output=dict(out=out, final_state=h.float()))
    return fixtures
