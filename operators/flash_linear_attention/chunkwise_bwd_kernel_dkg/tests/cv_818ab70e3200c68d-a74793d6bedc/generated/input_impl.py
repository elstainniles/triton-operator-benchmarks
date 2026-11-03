"""Existing independent CPU equations; file-generation main omitted."""
import torch
import math

def make_inputs():
    torch.manual_seed(20261031)
    B = 2
    T = 512
    G = 1
    K = 64
    H = 8
    V = 32
    L = 7
    BT = 32
    device = torch.device('cpu')
    dtype = torch.float32
    NT = ceil_div(T, BT)
    dh = torch.randn(B, NT, H, K, V, dtype=dtype, device=device)
    k = torch.randn(B, T, G, K, dtype=dtype, device=device)
    v = torch.randn(B, T, H, V, dtype=dtype, device=device)
    g = torch.randn(B, T, H, dtype=dtype, device=device)
    dg_last = torch.randn(B, NT, H, dtype=torch.float32, device=device)
    dk = torch.zeros(B, T, H, K, dtype=dtype, device=device)
    dg = torch.zeros(B, T, H, dtype=torch.float32, device=device)
    cu_seqlens = None
    inputs_dict = {'dh': dh.cpu(), 'k': k.cpu(), 'v': v.cpu(), 'g': g.cpu(), 'dg_last': dg_last.cpu(), 'dk': dk.cpu(), 'dg': dg.cpu(), 'cu_seqlens': cu_seqlens, 'B': B, 'T': T, 'G': G, 'K': K, 'H': H, 'V': V, 'L': L, 'BT': BT, 'NT': NT}
    dh_clone = dh.clone()
    k_clone = k.clone()
    v_clone = v.clone()
    g_clone = g.clone()
    dg_last_clone = dg_last.clone()
    dk_clone = dk.clone()
    dg_clone = dg.clone()
    grid_t = (NT, B * H)
    _prefix_values = locals().copy()
    if not (any((isinstance(_prefix_values.copy().get(key), dict) and _prefix_values.copy().get(key) for key in ('inputs_dict', 'input_dict', 'input_data'))) or (isinstance(_prefix_values.copy().get('data'), dict) and _prefix_values.copy()['data'].get('input_data'))):
        try:
            inputs_dict = {'dh': dh.cpu(), 'k': k.cpu(), 'v': v.cpu(), 'g': g.cpu(), 'dg_last': dg_last.cpu(), 'dk': dk.cpu(), 'dg': dg.cpu(), 'cu_seqlens': cu_seqlens, 'B': B, 'T': T, 'G': G, 'K': K, 'H': H, 'V': V, 'L': L, 'BT': BT, 'NT': NT}
        except NameError:
            pass
    return locals()

def ceil_div(x: int, y: int) -> int:
    return math.ceil(x / y)
