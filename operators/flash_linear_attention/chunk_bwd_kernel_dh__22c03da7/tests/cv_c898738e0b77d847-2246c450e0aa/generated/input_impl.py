"""Existing independent CPU equations; file-generation main omitted."""
import torch

def make_inputs():
    torch.manual_seed(42)
    B = 2
    T = 512
    H = 8
    K = 64
    V = 64
    HQ = H
    chunk_size = 64
    scale = 1.0
    dtype = torch.float32
    device = 'cpu'
    torch.manual_seed(42)
    q = torch.randn(B, T, HQ, K, dtype=dtype, device=device)
    k = torch.randn(B, T, H, K, dtype=dtype, device=device)
    v = torch.randn(B, T, H, V, dtype=dtype, device=device)
    do = torch.randn(B, T, HQ, V, dtype=dtype, device=device)
    h0 = torch.randn(B, H, K, V, dtype=torch.float32, device=device)
    dht = torch.randn(B, H, K, V, dtype=torch.float32, device=device)
    data = {'input_data': {'q': q, 'k': k, 'v': v, 'do': do, 'h0': h0, 'dht': dht, 'B': B, 'T': T, 'H': H, 'K': K, 'V': V, 'HQ': HQ, 'chunk_size': chunk_size, 'scale': scale}, 'gpu_output': {'dh': torch.zeros(B, (T + chunk_size - 1) // chunk_size, HQ, K, V, dtype=dtype, device=device), 'dh0': torch.zeros_like(h0)}}
    _prefix_values = locals().copy()
    if not (any((isinstance(_prefix_values.copy().get(key), dict) and _prefix_values.copy().get(key) for key in ('inputs_dict', 'input_dict', 'input_data'))) or (isinstance(_prefix_values.copy().get('data'), dict) and _prefix_values.copy()['data'].get('input_data'))):
        try:
            data = {'input_data': {'q': q, 'k': k, 'v': v, 'do': do, 'h0': h0, 'dht': dht, 'B': B, 'T': T, 'H': H, 'K': K, 'V': V, 'HQ': HQ, 'chunk_size': chunk_size, 'scale': scale}}
        except NameError:
            pass
    return locals()
