"""Existing independent CPU equations; file-generation main omitted."""
import types
triton = types.SimpleNamespace(cdiv=lambda n, d: (n + d - 1) // d, next_power_of_2=lambda n: 1 << (int(n) - 1).bit_length())
import torch

def make_inputs():
    torch.manual_seed(20261031)
    B, T, H, K, V = (4, 512, 8, 64, 128)
    chunk_size = 64
    device = torch.device('cpu')
    dtype = torch.float32
    q = torch.randn(B, T, H, K, dtype=dtype, device=device)
    k = torch.randn(B, T, H, K, dtype=dtype, device=device)
    v = torch.randn(B, T, H, V, dtype=dtype, device=device)
    v_new = torch.randn(B, T, H, V, dtype=dtype, device=device)
    b = torch.randn(B, T, H, K, dtype=dtype, device=device)
    NT = triton.cdiv(T, chunk_size)
    h = torch.randn(B, NT, H, K, V, dtype=dtype, device=device)
    inputs_dict = {'q': q.cpu(), 'k': k.cpu(), 'v': v.cpu(), 'v_new': v_new.cpu(), 'b': b.cpu(), 'h': h.cpu(), 'scale': K ** (-0.5), 'chunk_size': chunk_size, 'B': B, 'T': T, 'H': H, 'K': K, 'V': V}
    q_clone = q.clone()
    k_clone = k.clone()
    v_clone = v.clone()
    v_new_clone = v_new.clone()
    b_clone = b.clone()
    h_clone = h.clone()
    _prefix_values = locals().copy()
    if not (any((isinstance(_prefix_values.copy().get(key), dict) and _prefix_values.copy().get(key) for key in ('inputs_dict', 'input_dict', 'input_data'))) or (isinstance(_prefix_values.copy().get('data'), dict) and _prefix_values.copy()['data'].get('input_data'))):
        try:
            inputs_dict = {'q': q.cpu(), 'k': k.cpu(), 'v': v.cpu(), 'v_new': v_new.cpu(), 'b': b.cpu(), 'h': h.cpu(), 'scale': K ** (-0.5), 'chunk_size': chunk_size, 'B': B, 'T': T, 'H': H, 'K': K, 'V': V}
        except NameError:
            pass
    return locals()
