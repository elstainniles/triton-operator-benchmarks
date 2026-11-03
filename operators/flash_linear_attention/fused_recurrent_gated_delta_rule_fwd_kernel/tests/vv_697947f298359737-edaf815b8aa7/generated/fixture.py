"""Seeded CPU inputs and independent references, ported from the existing FLA test generator.
Logical fixture names are dictionary keys; this module reads and writes no data files.
Source SHA256: 146be15ebf8d7327e6769d15f01af2353d6d4b9978737cf529d245c31bed6336
Operation: fused_recurrent_gated_delta_test_fwd
"""
def make_fixtures(seed=20261030):
    import torch
    from pathlib import PurePosixPath
    SEED = seed
    N = 65539
    selected = op = 'fused_recurrent_gated_delta_test_fwd'
    fixtures = {}
    folder = PurePosixPath('') / selected
    gen = torch.Generator(device='cpu').manual_seed(SEED)
    shape = (2, 512, 4, 64)
    q = torch.randn(shape, generator=gen) * 0.01
    k = torch.randn(shape, generator=gen) * 0.01
    v = torch.randn(shape, generator=gen) * 0.01
    g = torch.nn.functional.logsigmoid(torch.rand((2, 512, 4), generator=gen))
    beta = torch.rand((2, 512, 4), generator=gen).sigmoid()
    h0 = torch.randn((2, 4, 64, 64), generator=gen) * 0.01
    h = h0.double()
    out = torch.empty_like(v)
    for t in range(512):
        qt = q[:, t].double()
        kt = k[:, t].double()
        vt = v[:, t].double()
        h = h * g[:, t].double().exp()[:, :, None, None]
        corrected = (vt - torch.einsum('bhk,bhkv->bhv', kt, h)) * beta[:, t].double()[:, :, None]
        h = h + kt[:, :, :, None] * corrected[:, :, None, :]
        out[:, t] = torch.einsum('bhk,bhkv->bhv', qt, h).float()
    inp = dict(q=q, k=k, v=v, g=g, beta=beta, h0=h0, o=torch.zeros_like(v), final_state=torch.zeros_like(h0), B=2, T=512, H=4, HV=4, K=64, V=64, BK=64, BV=8, NV=8, scale=1.0)
    path = folder / 'test_fused_recurrent_gated_delta_rule_fwd_kernel_cuda.pt'
    fixtures[path.name] = dict(input_data=inp, gpu_output=dict(o=out, final_state=h.float()))
    return fixtures
