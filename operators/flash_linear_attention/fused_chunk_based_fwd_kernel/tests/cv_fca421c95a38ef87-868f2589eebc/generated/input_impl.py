"""Existing independent CPU equations; file-generation main omitted."""
import os
import torch

def make_inputs():
    torch.manual_seed(42)
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    B, H, T, K, V = (4, 8, 2048, 16, 256)
    dtype = torch.float32
    device = 'cpu'
    q = torch.randn((B, H, T, K), dtype=dtype, device=device).contiguous()
    k = torch.randn((B, H, T, K), dtype=dtype, device=device).contiguous()
    v = torch.randn((B, H, T, V), dtype=dtype, device=device).contiguous()
    scale = K ** (-0.5)
    data = {}
    data['input_data'] = {'q': q.cpu(), 'k': k.cpu(), 'v': v.cpu(), 'scale': scale, 'B': B, 'H': H, 'T': T, 'K': K, 'V': V}
    _prefix_values = locals().copy()
    if not (any((isinstance(_prefix_values.copy().get(key), dict) and _prefix_values.copy().get(key) for key in ('inputs_dict', 'input_dict', 'input_data'))) or (isinstance(_prefix_values.copy().get('data'), dict) and _prefix_values.copy()['data'].get('input_data'))):
        try:
            data = {}
        except NameError:
            pass
        try:
            data['input_data'] = {'q': q.cpu(), 'k': k.cpu(), 'v': v.cpu(), 'scale': scale, 'B': B, 'H': H, 'T': T, 'K': K, 'V': V}
        except NameError:
            pass
    return locals()
