"""Existing independent CPU equations; file-generation main omitted."""
import torch
from recipe_algebraic_control import algebraic_control
from recipe_reference import reference

def make_fixtures():
    torch.set_num_threads(4)
    gen = torch.Generator(device='cpu').manual_seed(20261031)
    B, T, H, K, V, BT = (2, 512, 8, 64, 64, 32)
    inputs = {n: torch.randn((B, T, H, K), generator=gen, dtype=torch.float32) for n in ('q', 'k', 'v', 'g', 'do')}
    inputs.update({n: torch.randn((B * (T // BT), H, K, V), generator=gen, dtype=torch.float32) for n in ('h', 'dh')})
    inputs.update({n: torch.randn((B, T, H, K), generator=gen, dtype=torch.float32) for n in ('dq', 'dk')})
    inputs.update(scale=K ** (-0.5), chunk_size=BT)
    expected = reference(inputs, BT)
    control = algebraic_control(inputs, BT)
    checks = {}
    for key, value in expected.items():
        torch.testing.assert_close(value, control[key], rtol=1e-11, atol=1e-10, equal_nan=False)
        checks[key] = {'max_absolute_difference_between_two_independent_CPU_derivations': float((value - control[key]).abs().max()), 'shape': list(value.shape)}
    return {'input.pt': inputs, 'reference.pt': {k: v.float() for k, v in expected.items()}}
