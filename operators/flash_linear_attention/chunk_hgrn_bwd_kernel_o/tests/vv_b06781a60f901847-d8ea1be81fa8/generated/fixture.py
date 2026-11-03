"""Seeded CPU inputs and independent references, ported from the existing FLA test generator.
Logical fixture names are dictionary keys; this module reads and writes no data files.
Source SHA256: 146be15ebf8d7327e6769d15f01af2353d6d4b9978737cf529d245c31bed6336
Operation: chunk_hgrn_bwd_o
"""
def make_fixtures(seed=20261030):
    import torch
    from pathlib import PurePosixPath
    SEED = seed
    N = 65539
    selected = op = 'chunk_hgrn_bwd_o'
    fixtures = {}
    folder = PurePosixPath('') / selected
    gen = torch.Generator(device='cpu').manual_seed(SEED)
    x = torch.randn((2, 1024, 256), generator=gen)
    g = torch.nn.functional.logsigmoid(torch.randn((2, 1024, 256), generator=gen))
    h0 = torch.randn((2, 256), generator=gen)
    do = torch.randn(x.shape, generator=gen)
    o = torch.empty_like(x)
    gc = torch.empty_like(x)
    dx = torch.empty_like(x)
    dx_out = torch.empty_like(x)
    local = h0.double()
    for t in range(1024):
        if t and t % 128 == 0:
            local = torch.zeros_like(local)
        local = g[:, t].double().exp() * local + x[:, t].double()
        o[:, t] = local.float()
    full = torch.zeros((2, 256), dtype=torch.float64)
    local = full.clone()
    suffix = full.clone()
    for t in range(1023, -1, -1):
        if t % 64 == 63:
            local.zero_()
            suffix = g[:, t + 1].double().clone() if t + 1 < 1024 else torch.zeros_like(suffix)
        gc[:, t] = suffix.float()
        gt = g[:, t].double()
        local = local + do[:, t].double()
        dx[:, t] = local.float()
        local *= gt.exp()
        suffix += gt
        full = full + do[:, t].double()
        dx_out[:, t] = full.float()
        full *= gt.exp()
    previous = torch.cat((torch.zeros_like(o[:, :1]), o[:, :-1]), dim=1)
    dg = (previous.double() * dx_out.double() * g.double().exp()).float()
    for name, obj in [('x.pt', x), ('g.pt', g), ('h0.pt', h0), ('do.pt', do), ('o.pt', o), ('gc.pt', gc), ('dx.pt', dx), ('dx_out.pt', dx_out), ('dg.pt', dg)]:
        path = folder / name
        fixtures[path.name] = obj
    return fixtures
