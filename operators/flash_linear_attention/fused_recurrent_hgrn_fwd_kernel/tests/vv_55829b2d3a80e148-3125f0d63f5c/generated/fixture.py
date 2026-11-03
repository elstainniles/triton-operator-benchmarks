"""Seeded CPU inputs and independent references, ported from the existing FLA test generator.
Logical fixture names are dictionary keys; this module reads and writes no data files.
Source SHA256: 146be15ebf8d7327e6769d15f01af2353d6d4b9978737cf529d245c31bed6336
Operation: fused_recurrent_hgrn_fwd
"""
def make_fixtures(seed=20261030):
    import torch
    from pathlib import PurePosixPath
    SEED = seed
    N = 65539
    selected = op = 'fused_recurrent_hgrn_fwd'
    fixtures = {}
    folder = PurePosixPath('') / selected
    gen = torch.Generator(device='cpu').manual_seed(SEED)
    x = torch.randn((4, 2048, 512), generator=gen)
    g = torch.nn.functional.logsigmoid(torch.randn((4, 2048, 512), generator=gen))
    h0 = torch.randn((4, 512), generator=gen)
    o = torch.empty_like(x)
    h = h0.double()
    for t in range(2048):
        h = g[:, t].double().exp() * h + x[:, t].double()
        o[:, t] = h.float()
    for name, obj in [('inputs.pt', dict(x=x, g=g, h0=h0, output_final_state=True)), ('outputs.pt', dict(o=o, ht=h.float()))]:
        path = folder / name
        fixtures[path.name] = obj
    return fixtures
