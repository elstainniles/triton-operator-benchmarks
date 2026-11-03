"""Seeded CPU inputs and independent references, ported from the existing FLA test generator.
Logical fixture names are dictionary keys; this module reads and writes no data files.
Source SHA256: 146be15ebf8d7327e6769d15f01af2353d6d4b9978737cf529d245c31bed6336
Operation: fused_recurrent_hgrn_bwd
"""
def make_fixtures(seed=20261030):
    import torch
    from pathlib import PurePosixPath
    SEED = seed
    N = 65539
    selected = op = 'fused_recurrent_hgrn_bwd'
    fixtures = {}
    folder = PurePosixPath('') / selected
    gen = torch.Generator(device='cpu').manual_seed(SEED)
    x = torch.randn((4, 2048, 512), generator=gen)
    g = torch.nn.functional.logsigmoid(torch.randn((4, 2048, 512), generator=gen))
    h0 = torch.randn((4, 512), generator=gen)
    do = torch.randn(x.shape, generator=gen)
    dht = torch.randn((4, 512), generator=gen)
    o = torch.empty_like(x)
    h = h0.double()
    for t in range(2048):
        h = g[:, t].double().exp() * h + x[:, t].double()
        o[:, t] = h.float()
    dx = torch.empty_like(x)
    dg = torch.empty_like(x)
    d = dht.double()
    for t in range(2047, -1, -1):
        d = d + do[:, t].double()
        dx[:, t] = d.float()
        d = d * g[:, t].double().exp()
        dg[:, t] = (d * (o[:, t - 1].double() if t else h0.double())).float()
    for name, obj in [('inputs.pt', dict(g=g, o=o, do=do, dht=dht, h0=h0)), ('outputs.pt', dict(dx=dx, dg=dg, dh0=d.float()))]:
        path = folder / name
        fixtures[path.name] = obj
    return fixtures
