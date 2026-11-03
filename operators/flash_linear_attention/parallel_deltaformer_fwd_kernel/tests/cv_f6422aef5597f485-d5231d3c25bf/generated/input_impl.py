"""Existing independent CPU equations; file-generation main omitted."""
import math
import types
triton = types.SimpleNamespace(cdiv=lambda n, d: (n + d - 1) // d, next_power_of_2=lambda n: 1 << (int(n) - 1).bit_length())
import torch

def make_inputs():
    torch.manual_seed(42)
    torch.manual_seed(42)
    B, T, H, D = (4, 1024, 4, 128)
    C = min(512, T)
    dtype = torch.float32
    q = torch.randn(C, H, D, dtype=dtype)
    k = torch.randn(T, H, D, dtype=dtype)
    v = torch.randn(C, H, D, dtype=dtype)
    u = torch.randn(T, H, D, dtype=dtype)
    beta = torch.randn(C, H, dtype=dtype).sigmoid()
    qk_scale = 1.0 / math.sqrt(D)
    w = torch.zeros(C, H, C, dtype=dtype)
    lse = torch.zeros(C, H, dtype=torch.float)
    BLOCK_C = 64
    BLOCK_T = 64
    input_data = {'q': q, 'k': k, 'v': v, 'u': u, 'beta': beta, 'w': w, 'lse': lse, 'H': H, 'T': T, 'C': C, 'D': D, 'qk_scale': qk_scale, 'BLOCK_C': BLOCK_C, 'BLOCK_T': BLOCK_T}
    gpu_output = {'w': w.clone(), 'lse': lse.clone()}
    grid = (triton.cdiv(C, BLOCK_C), H)
    _prefix_values = locals().copy()
    if not (any((isinstance(_prefix_values.copy().get(key), dict) and _prefix_values.copy().get(key) for key in ('inputs_dict', 'input_dict', 'input_data'))) or (isinstance(_prefix_values.copy().get('data'), dict) and _prefix_values.copy()['data'].get('input_data'))):
        try:
            input_data = {'q': q, 'k': k, 'v': v, 'u': u, 'beta': beta, 'w': w, 'lse': lse, 'H': H, 'T': T, 'C': C, 'D': D, 'qk_scale': qk_scale, 'BLOCK_C': BLOCK_C, 'BLOCK_T': BLOCK_T}
        except NameError:
            pass
    return locals()
