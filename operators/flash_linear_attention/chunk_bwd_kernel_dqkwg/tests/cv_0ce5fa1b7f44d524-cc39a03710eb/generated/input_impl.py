"""Existing independent CPU equations; file-generation main omitted."""
import torch

def make_inputs():
    torch.manual_seed(42)
    B = 4
    T = 1024
    H = 8
    K = 128
    V = 128
    HV = H
    chunk_size = 64
    device = 'cpu'
    dtype = torch.float32
    torch.manual_seed(42)
    q = torch.randn(B, T, H, K, dtype=dtype, device=device)
    k = torch.randn(B, T, H, K, dtype=dtype, device=device)
    v = torch.randn(B, T, HV, V, dtype=dtype, device=device)
    do = torch.randn(B, T, HV, V, dtype=dtype, device=device)
    NT = (T + chunk_size - 1) // chunk_size
    h = torch.randn(B * NT, HV, K, V, dtype=dtype, device=device)
    dh = torch.randn(B * NT, HV, K, V, dtype=dtype, device=device)
    w = None
    g = None
    g_gamma = None
    dv = None
    scale = K ** (-0.5)
    dq = torch.zeros_like(q, dtype=dtype, device=device)
    dk = torch.zeros_like(k, dtype=dtype, device=device)
    dw = None
    dg = None
    data = {'input_data': {'q': q, 'k': k, 'v': v, 'do': do, 'h': h, 'dh': dh, 'w': w, 'g': g, 'g_gamma': g_gamma, 'dv': dv, 'scale': scale, 'B': B, 'T': T, 'H': H, 'HV': HV, 'K': K, 'V': V, 'chunk_size': chunk_size}, 'gpu_output': {'dq': dq, 'dk': dk, 'dw': dw, 'dg': dg}}
    _prefix_values = locals().copy()
    if not (any((isinstance(_prefix_values.copy().get(key), dict) and _prefix_values.copy().get(key) for key in ('inputs_dict', 'input_dict', 'input_data'))) or (isinstance(_prefix_values.copy().get('data'), dict) and _prefix_values.copy()['data'].get('input_data'))):
        try:
            data = {'input_data': {'q': q, 'k': k, 'v': v, 'do': do, 'h': h, 'dh': dh, 'w': w, 'g': g, 'g_gamma': g_gamma, 'dv': dv, 'scale': scale, 'B': B, 'T': T, 'H': H, 'HV': HV, 'K': K, 'V': V, 'chunk_size': chunk_size}}
        except NameError:
            pass
    return locals()
