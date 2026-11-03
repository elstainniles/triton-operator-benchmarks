"""Existing independent CPU equations; file-generation main omitted."""
import torch

def make_inputs():
    torch.manual_seed(20261031)
    B = 2
    T = 2048
    H = 8
    K = 128
    HV = H
    device = torch.device('cpu')
    dtype = torch.float16
    k = torch.randn(B, T, H, K, dtype=dtype, device=device) * (1.0 / K ** 0.5)
    v = torch.randn(B, T, HV, K, dtype=dtype, device=device)
    beta = torch.rand(B, T, HV, dtype=dtype, device=device).sigmoid()
    g = None
    chunk_size = 64
    data = {}
    data['input_data'] = {'k': k.cpu(), 'v': v.cpu(), 'g': g, 'beta': beta.cpu(), 'chunk_size': chunk_size, 'B': B, 'T': T, 'H': H, 'K': K, 'HV': HV, 'dtype': dtype}
    _prefix_values = locals().copy()
    if not (any((isinstance(_prefix_values.copy().get(key), dict) and _prefix_values.copy().get(key) for key in ('inputs_dict', 'input_dict', 'input_data'))) or (isinstance(_prefix_values.copy().get('data'), dict) and _prefix_values.copy()['data'].get('input_data'))):
        try:
            data = {}
        except NameError:
            pass
        try:
            data['input_data'] = {'k': k.cpu(), 'v': v.cpu(), 'g': g, 'beta': beta.cpu(), 'chunk_size': chunk_size, 'B': B, 'T': T, 'H': H, 'K': K, 'HV': HV, 'dtype': dtype}
        except NameError:
            pass
    return locals()
