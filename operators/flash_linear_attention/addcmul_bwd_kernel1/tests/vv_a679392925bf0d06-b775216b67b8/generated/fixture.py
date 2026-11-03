"""Fixed-seed CPU inputs and independent references; no external tensor files."""
import torch

SEED = 20261030
SHAPE = (4, 67, 33)
NAMES = ('r', 'w', 'k', 'v', 'a', 'g')

def data():
    gen = torch.Generator().manual_seed(SEED)
    rand = lambda shape: torch.randn(shape, generator=gen, dtype=torch.float32)
    hidden, delta = (rand(SHAPE), rand(SHAPE))
    weights = {f'x_{n}': rand((1, 1, SHAPE[-1])) for n in NAMES}
    grads = {f'd_x{n}': rand(SHAPE) for n in NAMES}
    fwd = {'hidden_states': hidden, 'delta': delta, **weights}
    fref = {f'ox{n}': (hidden.double() + delta.double() * weights[f'x_{n}'].double()).float() for n in NAMES}
    bwd = {**grads, **weights, 'hidden_states': hidden, 'delta': delta, 'use_xg': True}
    gh = sum((grads[f'd_x{n}'].double() for n in NAMES), torch.zeros(SHAPE, dtype=torch.float64)).float()
    gd = sum((grads[f'd_x{n}'].double() * weights[f'x_{n}'].double() for n in NAMES), torch.zeros(SHAPE, dtype=torch.float64)).float()
    return {'input_torch.pt': fwd, 'output_torch.pt': fref, 'test_addcmul_bwd_kernel1_cuda.pt': {'input_data': bwd, 'gpu_output': {'g_hiddn': gh, 'g_delta': gd}}}

def make_fixtures():
    return data()
