"""Existing independent CPU equations; file-generation main omitted."""
import torch

def make_inputs():
    torch.manual_seed(20261031)
    '\n    Test chunk_fwd_kernel_h with large shape\n    '
    B = 4
    T = 1024
    H = 8
    K = 128
    V = 128
    chunk_size = 64
    device = torch.device('cpu')
    dtype = torch.float16
    k = torch.randn(B, T, H, K, dtype=dtype, device=device)
    v = torch.randn(B, T, H, V, dtype=dtype, device=device)
    g = None
    g_gamma = None
    gk = None
    gv = None
    h0 = None
    output_final_state = False
    data = {}
    data['input_data'] = {'k': k.cpu(), 'v': v.cpu(), 'g': g, 'g_gamma': g_gamma, 'gk': gk, 'gv': gv, 'h0': h0, 'output_final_state': output_final_state, 'chunk_size': chunk_size, 'B': B, 'T': T, 'H': H, 'K': K, 'V': V}
    _prefix_values = locals().copy()
    if not (any((isinstance(_prefix_values.copy().get(key), dict) and _prefix_values.copy().get(key) for key in ('inputs_dict', 'input_dict', 'input_data'))) or (isinstance(_prefix_values.copy().get('data'), dict) and _prefix_values.copy()['data'].get('input_data'))):
        try:
            data = {}
        except NameError:
            pass
        try:
            data['input_data'] = {'k': k.cpu(), 'v': v.cpu(), 'g': g, 'g_gamma': g_gamma, 'gk': gk, 'gv': gv, 'h0': h0, 'output_final_state': output_final_state, 'chunk_size': chunk_size, 'B': B, 'T': T, 'H': H, 'K': K, 'V': V}
        except NameError:
            pass
    return locals()
