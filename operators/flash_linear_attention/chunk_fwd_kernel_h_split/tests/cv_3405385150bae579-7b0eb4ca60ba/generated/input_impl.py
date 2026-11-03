"""Existing independent CPU equations; file-generation main omitted."""
import torch

def make_inputs():
    torch.manual_seed(42)
    B = 2
    T = 2048
    H = 8
    K = 128
    V = 128
    dtype = torch.float32
    device = 'cpu'
    torch.manual_seed(42)
    k = torch.randn(B, T, H, K, dtype=dtype, device=device)
    v = torch.randn(B, T, H, V, dtype=dtype, device=device)
    g = None
    gk = None
    gv = None
    h0 = None
    chunk_size = 64
    split_size = 256
    output_final_state = True
    data = {'input_data': {'k': k, 'v': v, 'g': g, 'gk': gk, 'gv': gv, 'h0': h0, 'B': B, 'T': T, 'H': H, 'K': K, 'V': V, 'chunk_size': chunk_size, 'split_size': split_size, 'output_final_state': output_final_state}, 'gpu_output': {'hr': torch.zeros(B, H, K, V, dtype=dtype, device=device), 'ht': torch.zeros(B, H, K, V, dtype=dtype, device=device)}}
    _prefix_values = locals().copy()
    if not (any((isinstance(_prefix_values.copy().get(key), dict) and _prefix_values.copy().get(key) for key in ('inputs_dict', 'input_dict', 'input_data'))) or (isinstance(_prefix_values.copy().get('data'), dict) and _prefix_values.copy()['data'].get('input_data'))):
        try:
            data = {'input_data': {'k': k, 'v': v, 'g': g, 'gk': gk, 'gv': gv, 'h0': h0, 'B': B, 'T': T, 'H': H, 'K': K, 'V': V, 'chunk_size': chunk_size, 'split_size': split_size, 'output_final_state': output_final_state}}
        except NameError:
            pass
    return locals()
