"""Existing independent CPU equations; file-generation main omitted."""
import types
triton = types.SimpleNamespace(cdiv=lambda n, d: (n + d - 1) // d, next_power_of_2=lambda n: 1 << (int(n) - 1).bit_length())
import torch.nn.functional as F
import torch

def make_inputs():
    torch.manual_seed(42)
    B, T, H, D = (4, 2048, 8, 128)
    chunk_size = 64
    dtype = torch.float32
    scale = 0.1
    torch.manual_seed(42)
    q = torch.randn(B, T, H, D, dtype=dtype)
    k = torch.randn(B, T, H, D, dtype=dtype)
    do = torch.randn(B, T, H, D, dtype=dtype)
    NT = triton.cdiv(T, chunk_size)
    dh = torch.randn(B, NT, H, D, D, dtype=torch.float32)
    g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.float32))
    data = {'input_data': {'q': q, 'k': k, 'do': do, 'dh': dh, 'g': g, 'B': B, 'T': T, 'H': H, 'D': D, 'chunk_size': chunk_size, 'scale': scale}, 'gpu_output': {'dv': torch.zeros_like(do)}}
    _prefix_values = locals().copy()
    if not (any((isinstance(_prefix_values.copy().get(key), dict) and _prefix_values.copy().get(key) for key in ('inputs_dict', 'input_dict', 'input_data'))) or (isinstance(_prefix_values.copy().get('data'), dict) and _prefix_values.copy()['data'].get('input_data'))):
        try:
            data = {'input_data': {'q': q, 'k': k, 'do': do, 'dh': dh, 'g': g, 'B': B, 'T': T, 'H': H, 'D': D, 'chunk_size': chunk_size, 'scale': scale}}
        except NameError:
            pass
    return locals()
