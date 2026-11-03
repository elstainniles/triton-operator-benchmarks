"""Existing independent CPU equations; file-generation main omitted."""
import torch

def make_inputs():
    torch.manual_seed(42)
    torch.manual_seed(42)
    B, T, H, D = (2, 2048, 8, 64)
    dtype = torch.float32
    device = 'cpu'
    scale = D ** (-0.5)
    q = torch.randn((B, T, H, D), dtype=dtype, device=device) * 0.1
    k = torch.randn((B, T, H, D), dtype=dtype, device=device) * 0.1
    w = torch.randn((B, T, H, D), dtype=dtype, device=device) * 0.1
    g = None
    h0 = torch.randn((B, H, D, D), dtype=torch.float32, device=device) * 0.01
    dht = torch.randn((B, H, D, D), dtype=torch.float32, device=device) * 0.01
    do = torch.randn((B, T, H, D), dtype=dtype, device=device) * 0.1
    dv = torch.randn((B, T, H, D), dtype=dtype, device=device) * 0.1
    data = {'input_data': {'q': q, 'k': k, 'w': w, 'g': g, 'h0': h0, 'dht': dht, 'do': do, 'dv': dv, 'scale': scale, 'chunk_size': 64}, 'gpu_output': {'dh': torch.zeros(B, (T + 63) // 64, H, D, D, dtype=dtype, device=device), 'dh0': torch.zeros_like(h0), 'dv2': torch.zeros_like(dv)}}
    _prefix_values = locals().copy()
    if not (any((isinstance(_prefix_values.copy().get(key), dict) and _prefix_values.copy().get(key) for key in ('inputs_dict', 'input_dict', 'input_data'))) or (isinstance(_prefix_values.copy().get('data'), dict) and _prefix_values.copy()['data'].get('input_data'))):
        try:
            data = {'input_data': {'q': q, 'k': k, 'w': w, 'g': g, 'h0': h0, 'dht': dht, 'do': do, 'dv': dv, 'scale': scale, 'chunk_size': 64}}
        except NameError:
            pass
    return locals()
