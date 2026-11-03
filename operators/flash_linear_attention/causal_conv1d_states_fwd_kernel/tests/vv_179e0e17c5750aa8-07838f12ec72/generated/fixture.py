"""Seeded CPU inputs and independent references, ported from the existing FLA test generator.
Logical fixture names are dictionary keys; this module reads and writes no data files.
Source SHA256: 146be15ebf8d7327e6769d15f01af2353d6d4b9978737cf529d245c31bed6336
Operation: causal_conv1d_states_fwd
"""
def make_fixtures(seed=20261030):
    import torch
    from pathlib import PurePosixPath
    SEED = seed
    N = 65539
    selected = op = 'causal_conv1d_states_fwd'
    fixtures = {}
    folder = PurePosixPath('') / selected
    gen = torch.Generator(device='cpu').manual_seed(SEED)
    x = torch.randn((4, 1024, 1024), generator=gen, dtype=torch.float32)
    init = torch.randn((4, 1024, 4), generator=gen, dtype=torch.float32)
    output = x[:, -4:, :].transpose(1, 2).contiguous()
    for suffix, obj in [('_in.pt', dict(x=x, initial_state=init, state_len=4)), ('_out.pt', dict(output=output))]:
        path = folder / ('causal_conv1d_states_fwd_kernel' + suffix)
        fixtures[path.name] = obj
    return fixtures
