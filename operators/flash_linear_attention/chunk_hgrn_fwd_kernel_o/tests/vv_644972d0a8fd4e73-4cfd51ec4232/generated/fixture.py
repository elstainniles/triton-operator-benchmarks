"""Seeded CPU inputs and independent references, ported from the existing FLA test generator.
Logical fixture names are dictionary keys; this module reads and writes no data files.
Source SHA256: 146be15ebf8d7327e6769d15f01af2353d6d4b9978737cf529d245c31bed6336
Operation: chunk_hgrn_fwd_o
"""
def make_fixtures(seed=20261030):
    import torch
    from pathlib import PurePosixPath
    SEED = seed
    N = 65539
    selected = op = 'chunk_hgrn_fwd_o'
    fixtures = {}
    folder = PurePosixPath('') / selected
    gen = torch.Generator(device='cpu').manual_seed(SEED)
    x = torch.randn((4, 2048, 512), generator=gen)
    g = torch.nn.functional.logsigmoid(torch.randn((4, 2048, 512), generator=gen))
    h0 = torch.randn((4, 512), generator=gen)
    o = torch.empty_like(x)
    gc = torch.empty_like(g)
    out = torch.empty_like(x)
    h = h0.double()
    local = h0.double()
    prefix = torch.zeros_like(h)
    for t in range(2048):
        if t and t % 128 == 0:
            local = torch.zeros_like(local)
            prefix.zero_()
        gt = g[:, t, :].double()
        xt = x[:, t, :].double()
        h = gt.exp() * h + xt
        local = gt.exp() * local + xt
        prefix += gt
        o[:, t, :] = local.float()
        gc[:, t, :] = prefix.float()
        out[:, t, :] = h.float()
    for name, obj in [('x.pt', x), ('g.pt', g), ('h0.pt', h0), ('o.pt', o), ('gc.pt', gc), ('o_out.pt', out)]:
        path = folder / name
        fixtures[path.name] = obj
    return fixtures
