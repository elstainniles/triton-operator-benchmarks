"""Existing independent CPU equations; file-generation main omitted."""
import torch
import math

def make_inputs():
    torch.manual_seed(20261031)
    B = 2
    T = 1024
    H = 8
    K = 64
    V = 64
    L = 12
    BT = 64
    G = 1
    NT = ceil_div(T, BT)
    device = torch.device('cpu')
    dh = torch.randn(B, NT, H, K, V, dtype=torch.float32, device=device)
    k = torch.randn(B, T, G, K, dtype=torch.float32, device=device)
    g = torch.randn(B, T, H, dtype=torch.float32, device=device)
    dv = torch.zeros(B, T, H, V, dtype=torch.float32, device=device)
    inputs_dict = {'dh': dh.cpu(), 'k': k.cpu(), 'g': g.cpu(), 'dv': dv.cpu(), 'cu_seqlens': None, 'B': B, 'T': T, 'H': H, 'K': K, 'V': V, 'L': L, 'BT': BT, 'G': G, 'NT': NT}
    dh_clone = dh.clone()
    k_clone = k.clone()
    g_clone = g.clone()
    dv_clone = dv.clone()
    grid = (NT, B * H)
    _prefix_values = locals().copy()
    if not (any((isinstance(_prefix_values.copy().get(key), dict) and _prefix_values.copy().get(key) for key in ('inputs_dict', 'input_dict', 'input_data'))) or (isinstance(_prefix_values.copy().get('data'), dict) and _prefix_values.copy()['data'].get('input_data'))):
        try:
            inputs_dict = {'dh': dh.cpu(), 'k': k.cpu(), 'g': g.cpu(), 'dv': dv.cpu(), 'cu_seqlens': None, 'B': B, 'T': T, 'H': H, 'K': K, 'V': V, 'L': L, 'BT': BT, 'G': G, 'NT': NT}
        except NameError:
            pass
    return locals()

def ceil_div(x: int, y: int) -> int:
    return math.ceil(x / y)
