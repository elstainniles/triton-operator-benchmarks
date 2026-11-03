"""Existing independent CPU equations; file-generation main omitted."""
import torch.nn.functional as F
import torch

def make_inputs():
    torch.manual_seed(42)
    B, T, H, D = (4, 2048, 8, 128)
    chunk_size = 64
    dtype = torch.float32
    torch.manual_seed(42)
    k = torch.randn(B, T, H, D, dtype=dtype)
    w = torch.randn(B, T, H, D, dtype=dtype)
    u = torch.randn(B, T, H, D, dtype=dtype)
    g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.float32))
    data = {'input_data': {'k': k, 'w': w, 'u': u, 'g': g, 'initial_state': None, 'output_final_state': False, 'chunk_size': chunk_size}, 'gpu_output': {'h': torch.zeros(B, (T + chunk_size - 1) // chunk_size, H, D, D, dtype=dtype), 'v_new': torch.zeros_like(u), 'final_state': None}}
    _prefix_values = locals().copy()
    if not (any((isinstance(_prefix_values.copy().get(key), dict) and _prefix_values.copy().get(key) for key in ('inputs_dict', 'input_dict', 'input_data'))) or (isinstance(_prefix_values.copy().get('data'), dict) and _prefix_values.copy()['data'].get('input_data'))):
        try:
            data = {'input_data': {'k': k, 'w': w, 'u': u, 'g': g, 'initial_state': None, 'output_final_state': False, 'chunk_size': chunk_size}}
        except NameError:
            pass
    return locals()
