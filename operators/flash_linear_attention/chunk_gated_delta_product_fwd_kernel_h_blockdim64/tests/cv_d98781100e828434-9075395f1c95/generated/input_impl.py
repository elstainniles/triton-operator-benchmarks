"""Existing independent CPU equations; file-generation main omitted."""
import torch

def make_inputs():
    torch.manual_seed(42)
    torch.manual_seed(42)
    B, T, H, D = (2, 2048, 8, 64)
    num_householder = 2
    dtype = torch.float32
    device = 'cpu'
    k = torch.randn((B, T * num_householder, H, D), dtype=dtype, device=device) * 0.1
    w = torch.randn((B, T * num_householder, H, D), dtype=dtype, device=device) * 0.1
    u = torch.randn((B, T * num_householder, H, D), dtype=dtype, device=device) * 0.1
    g = None
    h0 = torch.randn((B, H, D, D), dtype=torch.float32, device=device) * 0.01
    data = {'input_data': {'k': k, 'w': w, 'u': u, 'g': g, 'initial_state': h0, 'output_final_state': True, 'num_householder': num_householder, 'chunk_size': 64, 'save_new_value': True}, 'gpu_output': {'h': torch.zeros(B, T, H, D, D, dtype=dtype, device=device), 'v_new': torch.zeros_like(u), 'final_state': torch.zeros(B, H, D, D, dtype=torch.float32, device=device)}}
    _prefix_values = locals().copy()
    if not (any((isinstance(_prefix_values.copy().get(key), dict) and _prefix_values.copy().get(key) for key in ('inputs_dict', 'input_dict', 'input_data'))) or (isinstance(_prefix_values.copy().get('data'), dict) and _prefix_values.copy()['data'].get('input_data'))):
        try:
            data = {'input_data': {'k': k, 'w': w, 'u': u, 'g': g, 'initial_state': h0, 'output_final_state': True, 'num_householder': num_householder, 'chunk_size': 64, 'save_new_value': True}}
        except NameError:
            pass
    return locals()
