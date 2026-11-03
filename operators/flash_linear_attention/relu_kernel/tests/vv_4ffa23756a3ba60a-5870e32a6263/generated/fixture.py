"""Seeded CPU inputs and independent references, ported from the existing FLA test generator.
Logical fixture names are dictionary keys; this module reads and writes no data files.
Source SHA256: 146be15ebf8d7327e6769d15f01af2353d6d4b9978737cf529d245c31bed6336
Operation: relu
"""
def make_fixtures(seed=20261030):
    import torch
    from pathlib import PurePosixPath
    SEED = seed
    N = 65539
    selected = op = 'relu'
    fixtures = {}
    folder = PurePosixPath('') / op
    gen = torch.Generator(device='cpu').manual_seed(SEED)
    n = 1048576 if op.startswith('softplus') else N
    x = torch.randn(n, generator=gen, dtype=torch.float32)
    ref = getattr(torch, op)(x)
    in_name = 'inputs.pt' if op == 'exp' else op + '_in.pt'
    out_name = 'outputs.pt' if op == 'exp' else op + '_out.pt'
    fixtures[(folder / in_name).name] = {'x': x}
    fixtures[(folder / out_name).name] = {'output': ref, 'baseline_output': x.clone()}
    return fixtures
