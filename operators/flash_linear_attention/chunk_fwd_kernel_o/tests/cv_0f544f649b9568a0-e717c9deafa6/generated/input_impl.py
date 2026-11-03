"""Existing independent CPU equations; file-generation main omitted."""
import types
triton = types.SimpleNamespace(cdiv=lambda n, d: (n + d - 1) // d, next_power_of_2=lambda n: 1 << (int(n) - 1).bit_length())
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
    dtype = torch.float32
    device = 'cpu'
    torch.manual_seed(42)
    q = torch.randn(B, T, H, K, dtype=dtype, device=device)
    k = torch.randn(B, T, H, K, dtype=dtype, device=device)
    v = torch.randn(B, T, HV, V, dtype=dtype, device=device)
    NT = triton.cdiv(T, chunk_size)
    h = torch.randn(B * NT, HV, K, V, dtype=dtype, device=device)
    g = None
    g_gamma = None
    scale = K ** (-0.5)
    data = {'input_data': {'q': q, 'k': k, 'v': v, 'h': h, 'g': g, 'g_gamma': g_gamma, 'scale': scale, 'cu_seqlens': None, 'chunk_size': chunk_size, 'transpose_state_layout': False}, 'gpu_output': {'o': torch.zeros_like(v)}}
    _prefix_values = locals().copy()
    if not (any((isinstance(_prefix_values.copy().get(key), dict) and _prefix_values.copy().get(key) for key in ('inputs_dict', 'input_dict', 'input_data'))) or (isinstance(_prefix_values.copy().get('data'), dict) and _prefix_values.copy()['data'].get('input_data'))):
        try:
            data = {'input_data': {'q': q, 'k': k, 'v': v, 'h': h, 'g': g, 'g_gamma': g_gamma, 'scale': scale, 'cu_seqlens': None, 'chunk_size': chunk_size, 'transpose_state_layout': False}}
        except NameError:
            pass
    return locals()
