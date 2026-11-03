"""Seeded CPU inputs and independent references, ported from the existing FLA test generator.
Logical fixture names are dictionary keys; this module reads and writes no data files.
Source SHA256: 146be15ebf8d7327e6769d15f01af2353d6d4b9978737cf529d245c31bed6336
Operation: chunk_hgrn_bwd_h
"""
def make_fixtures(seed=20261030):
    import torch
    from pathlib import PurePosixPath
    SEED = seed
    N = 65539
    selected = op = 'chunk_hgrn_bwd_h'
    fixtures = {}
    folder = PurePosixPath('') / selected
    gen = torch.Generator(device='cpu').manual_seed(SEED)
    x = torch.randn((4, 2048, 512), generator=gen)
    g = torch.nn.functional.logsigmoid(torch.randn((4, 2048, 512), generator=gen))
    h0 = torch.randn((4, 512), generator=gen)
    do = torch.randn(x.shape, generator=gen)
    gc = torch.empty_like(g)
    dx = torch.empty_like(g)
    local = torch.zeros((4, 512), dtype=torch.float64)
    suffix = local.clone()
    for t in range(2047, -1, -1):
        if t % 128 == 127:
            local.zero_()
            suffix = g[:, t + 1].double().clone() if t + 1 < 2048 else torch.zeros_like(suffix)
        gc[:, t] = suffix.float()
        gt = g[:, t].double()
        local += do[:, t].double()
        dx[:, t] = local.float()
        local *= gt.exp()
        suffix += gt
    for name, obj in [('x.pt', x), ('g.pt', g), ('h0.pt', h0), ('do.pt', do), ('gc.pt', gc), ('dx.pt', dx)]:
        path = folder / name
        fixtures[path.name] = obj
    return fixtures
