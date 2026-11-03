"""Existing independent CPU equations; file-generation main omitted."""
import types
triton = types.SimpleNamespace(cdiv=lambda n, d: (n + d - 1) // d, next_power_of_2=lambda n: 1 << (int(n) - 1).bit_length())
import torch

def make_inputs():
    torch.manual_seed(20261031)
    B = 2
    T = 256
    H = 8
    HQ = 16
    K = 32
    S = 64
    BT = 32
    BS = 32
    device = torch.device('cpu')
    dtype = torch.float32
    q = torch.randn(B, T, HQ, K, dtype=dtype, device=device)
    w1 = torch.randn(B, T, H, K, dtype=dtype, device=device)
    w2 = torch.randn(B, T, H, K, dtype=dtype, device=device)
    num_blocks = triton.cdiv(T, S)
    import os
    inputs_dict = {'q': q.cpu(), 'w1': w1.cpu(), 'w2': w2.cpu(), 'cu_seqlens': None, 'BT': BT, 'BS': BS, 'S': S}
    _prefix_values = locals().copy()
    if not (any((isinstance(_prefix_values.copy().get(key), dict) and _prefix_values.copy().get(key) for key in ('inputs_dict', 'input_dict', 'input_data'))) or (isinstance(_prefix_values.copy().get('data'), dict) and _prefix_values.copy()['data'].get('input_data'))):
        try:
            inputs_dict = {'q': q.cpu(), 'w1': w1.cpu(), 'w2': w2.cpu(), 'cu_seqlens': None, 'BT': BT, 'BS': BS, 'S': S}
        except NameError:
            pass
    return locals()
