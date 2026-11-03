"""Existing independent CPU equations; file-generation main omitted."""
import torch

def make_inputs():
    torch.manual_seed(42)
    torch.manual_seed(42)
    B = 2
    H = 8
    T = 1024
    K = 64
    V = 64
    BTL = 64
    BTS = 32
    BK = 64
    BV = 64
    dtype = torch.float32
    device = 'cpu'
    scale = K ** (-0.5)
    q = torch.randn(B, H, T, K, dtype=dtype, device=device)
    k = torch.randn(B, H, T, K, dtype=dtype, device=device)
    v = torch.randn(B, H, T, V, dtype=dtype, device=device)
    do = torch.randn(B, H, T, V, dtype=dtype, device=device)
    dz = torch.randn(B, H, T, dtype=dtype, device=device)
    data = {'input_data': {'q': q, 'k': k, 'v': v, 'do': do, 'dz': dz, 'scale': scale, 'B': B, 'H': H, 'T': T, 'K': K, 'V': V, 'BTL': BTL, 'BTS': BTS, 'BK': BK, 'BV': BV}, 'gpu_output': {'dq': torch.zeros(1, B, H, T, K, dtype=dtype, device=device)}}
    _prefix_values = locals().copy()
    if not (any((isinstance(_prefix_values.copy().get(key), dict) and _prefix_values.copy().get(key) for key in ('inputs_dict', 'input_dict', 'input_data'))) or (isinstance(_prefix_values.copy().get('data'), dict) and _prefix_values.copy()['data'].get('input_data'))):
        try:
            data = {'input_data': {'q': q, 'k': k, 'v': v, 'do': do, 'dz': dz, 'scale': scale, 'B': B, 'H': H, 'T': T, 'K': K, 'V': V, 'BTL': BTL, 'BTS': BTS, 'BK': BK, 'BV': BV}}
        except NameError:
            pass
    return locals()
