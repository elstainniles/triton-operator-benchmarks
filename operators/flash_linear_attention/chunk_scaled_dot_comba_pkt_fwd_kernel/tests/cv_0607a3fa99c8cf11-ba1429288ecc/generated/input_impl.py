"""Existing independent CPU equations; file-generation main omitted."""
import torch.nn.functional as F
import torch

def make_inputs():
    torch.manual_seed(42)
    B, T, H, K = (4, 2048, 8, 128)
    chunk_size = 64
    dtype = torch.float32
    device = 'cpu'
    torch.manual_seed(42)
    k = torch.randn(B, T, H, K, dtype=dtype, device=device)
    p = torch.randn(B, T, H, K, dtype=dtype, device=device)
    beta = torch.rand(B, T, H, dtype=dtype, device=device).sigmoid()
    g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.float32, device=device))
    g0 = g - F.logsigmoid(torch.rand(B, T, H, dtype=torch.float32, device=device))
    data = {'input_data': {'k': k, 'p': p, 'beta': beta, 'g': g, 'g0': g0, 'chunk_size': chunk_size}, 'gpu_output': {'A': torch.zeros(B, T, H, chunk_size, dtype=dtype, device=device)}}
    _prefix_values = locals().copy()
    if not (any((isinstance(_prefix_values.copy().get(key), dict) and _prefix_values.copy().get(key) for key in ('inputs_dict', 'input_dict', 'input_data'))) or (isinstance(_prefix_values.copy().get('data'), dict) and _prefix_values.copy()['data'].get('input_data'))):
        try:
            data = {'input_data': {'k': k, 'p': p, 'beta': beta, 'g': g, 'g0': g0, 'chunk_size': chunk_size}}
        except NameError:
            pass
    return locals()
