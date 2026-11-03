"""Existing independent CPU equations; file-generation main omitted."""
import torch

def make_inputs():
    torch.manual_seed(42)
    torch.manual_seed(42)
    B, T, H, HV, D = (2, 2048, 8, 8, 64)
    dtype = torch.float32
    device = 'cpu'
    scale = D ** (-0.5)
    q = torch.randn((B, T, H, D), dtype=dtype, device=device) * 0.1
    k = torch.randn((B, T, H, D), dtype=dtype, device=device) * 0.1
    v = torch.randn((B, T, HV, D), dtype=dtype, device=device) * 0.1
    do = torch.randn((B, T, HV, D), dtype=dtype, device=device) * 0.1
    A = torch.randn((B, T, HV, 64), dtype=dtype, device=device) * 0.1
    data = {'input_data': {'q': q, 'k': k, 'v': v, 'do': do, 'A': A, 'scale': scale, 'chunk_size': 64}, 'gpu_output': {'dA': torch.zeros_like(A), 'dv': torch.zeros_like(do)}}
    _prefix_values = locals().copy()
    if not (any((isinstance(_prefix_values.copy().get(key), dict) and _prefix_values.copy().get(key) for key in ('inputs_dict', 'input_dict', 'input_data'))) or (isinstance(_prefix_values.copy().get('data'), dict) and _prefix_values.copy()['data'].get('input_data'))):
        try:
            data = {'input_data': {'q': q, 'k': k, 'v': v, 'do': do, 'A': A, 'scale': scale, 'chunk_size': 64}}
        except NameError:
            pass
    return locals()
