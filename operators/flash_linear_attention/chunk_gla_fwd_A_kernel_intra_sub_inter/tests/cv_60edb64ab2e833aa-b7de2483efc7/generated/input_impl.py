"""Existing independent CPU equations; file-generation main omitted."""
import torch

def make_inputs():
    torch.manual_seed(20261031)
    B, T, H, K = (2, 2048, 8, 128)
    chunk_size = 64
    scale = K ** (-0.5)
    device = torch.device('cpu')
    dtype = torch.float32
    q = torch.randn(B, T, H, K, dtype=dtype, device=device)
    k = torch.randn(B, T, H, K, dtype=dtype, device=device)
    g = torch.randn(B, T, H, K, dtype=dtype, device=device)
    inputs_dict = {'q': q.cpu(), 'k': k.cpu(), 'g': g.cpu(), 'cu_seqlens': None, 'chunk_size': chunk_size, 'scale': scale}
    _prefix_values = locals().copy()
    if not (any((isinstance(_prefix_values.copy().get(key), dict) and _prefix_values.copy().get(key) for key in ('inputs_dict', 'input_dict', 'input_data'))) or (isinstance(_prefix_values.copy().get('data'), dict) and _prefix_values.copy()['data'].get('input_data'))):
        try:
            inputs_dict = {'q': q.cpu(), 'k': k.cpu(), 'g': g.cpu(), 'cu_seqlens': None, 'chunk_size': chunk_size, 'scale': scale}
        except NameError:
            pass
    return locals()
