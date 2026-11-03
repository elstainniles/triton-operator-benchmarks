"""Existing independent CPU equations; file-generation main omitted."""
import torch

def make_inputs():
    torch.manual_seed(42)
    B = 4
    T = 1024
    H = 8
    K = 128
    V = 128
    chunk_size = 64
    device = 'cpu'
    dtype = torch.float32
    torch.manual_seed(42)
    k = torch.randn(B, T, H, K, dtype=dtype, device=device)
    v = torch.randn(B, T, H, V, dtype=dtype, device=device)
    g = None
    gk = None
    gv = None
    h0 = None
    output_final_state = False
    NT = (T + chunk_size - 1) // chunk_size
    h = torch.zeros(B, NT, H, K, V, dtype=dtype, device=device)
    ht = None
    data = {'input_data': {'k': k, 'v': v, 'g': g, 'gk': gk, 'gv': gv, 'h0': h0, 'B': B, 'T': T, 'H': H, 'K': K, 'V': V, 'chunk_size': chunk_size, 'output_final_state': output_final_state}, 'gpu_output': {'h': h, 'ht': ht}}
    _prefix_values = locals().copy()
    if not (any((isinstance(_prefix_values.copy().get(key), dict) and _prefix_values.copy().get(key) for key in ('inputs_dict', 'input_dict', 'input_data'))) or (isinstance(_prefix_values.copy().get('data'), dict) and _prefix_values.copy()['data'].get('input_data'))):
        try:
            data = {'input_data': {'k': k, 'v': v, 'g': g, 'gk': gk, 'gv': gv, 'h0': h0, 'B': B, 'T': T, 'H': H, 'K': K, 'V': V, 'chunk_size': chunk_size, 'output_final_state': output_final_state}}
        except NameError:
            pass
    return locals()
