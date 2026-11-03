"""Seeded CPU inputs and independent references, ported from the existing FLA test generator.
Logical fixture names are dictionary keys; this module reads and writes no data files.
Source SHA256: 146be15ebf8d7327e6769d15f01af2353d6d4b9978737cf529d245c31bed6336
Operation: k_update_bwd_long
"""
def make_fixtures(seed=20261030):
    import torch
    from pathlib import PurePosixPath
    SEED = seed
    N = 65539
    selected = op = 'k_update_bwd_long'
    fixtures = {}
    folder = PurePosixPath('') / selected
    gen = torch.Generator(device='cpu').manual_seed(SEED)
    T = 512 if selected == 'k_update_bwd_short' else 4096
    shape = (4, T, 4096)
    k = torch.randn(shape, generator=gen, dtype=torch.float16)
    a = torch.randn(shape, generator=gen, dtype=torch.float16)
    go = torch.randn(shape, generator=gen, dtype=torch.float16)
    ka = torch.randn((4096,), generator=gen, dtype=torch.float16)
    dk = torch.zeros(shape, dtype=torch.float16)
    da = torch.zeros_like(dk)
    tmp = torch.zeros_like(dk)
    n = 4 * T * 64
    kk = k.view(-1)[:n].double().reshape(-1, 64)
    aa = a.view(-1)[:n].double().reshape(-1, 64)
    gg = go.view(-1)[:n].double().reshape(-1, 64)
    kv = ka[:64].double()
    out_dk = dk.clone()
    out_da = da.clone()
    out_tmp = tmp.clone()
    out_dk.view(-1)[:n] = (gg * (1 + (aa - 1) * kv)).reshape(-1).half()
    out_da.view(-1)[:n] = (gg * kk * kv).reshape(-1).half()
    out_tmp.view(-1)[:n] = (gg * kk * (aa - 1)).reshape(-1).half()
    out_dka = out_tmp.view(-1)[:n].reshape(-1, 4096).double().sum(0).half()
    inp = dict(grad_out=go, k=k, a=a, ka=ka, dk=dk, da=da, dka_tmp=tmp, B=4, T=T, H=64, D=64, BD=64)
    inp.update(BT=64, NT=64)
    path = folder / ('test_k_update_bwd_kernel_short_cuda.pt' if selected == 'k_update_bwd_short' else 'test_k_update_bwd_kernel_long_cuda.pt')
    fixtures[path.name] = dict(input_data=inp, gpu_output=dict(dk=out_dk, da=out_da, dka=out_dka))
    return fixtures
