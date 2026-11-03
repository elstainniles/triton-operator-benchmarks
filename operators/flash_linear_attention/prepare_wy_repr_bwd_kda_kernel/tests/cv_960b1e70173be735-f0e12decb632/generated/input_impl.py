"""Existing independent CPU equations; file-generation main omitted."""
import torch

def make_inputs():
    torch.manual_seed(20261031)
    '\n    Test prepare_wy_repr_bwd_kda_kernel with large shape configuration\n    '
    B = 2
    T = 1024
    H = 8
    HV = 8
    K = 64
    V = 64
    BT = 64
    device = torch.device('cpu')
    dtype = torch.float32
    k = torch.randn(B, T, H, K, device=device, dtype=dtype)
    v = torch.randn(B, T, HV, V, device=device, dtype=dtype)
    beta = torch.randn(B, T, HV, device=device, dtype=dtype)
    gk = torch.randn(B, T, HV, K, device=device, dtype=dtype)
    A = torch.randn(B, T, HV, BT, device=device, dtype=dtype)
    dk = torch.randn(B, T, HV, K, device=device, dtype=dtype)
    dw = torch.randn(B, T, HV, K, device=device, dtype=dtype)
    du = torch.randn(B, T, HV, V, device=device, dtype=dtype)
    dg = torch.randn(B, T, HV, K, device=device, dtype=dtype)
    import os
    inputs_dict = {'k': k.cpu(), 'v': v.cpu(), 'beta': beta.cpu(), 'gk': gk.cpu(), 'A': A.cpu(), 'dk': dk.cpu(), 'dw': dw.cpu(), 'du': du.cpu(), 'dg': dg.cpu(), 'cu_seqlens': None, 'chunk_indices': None}
    _prefix_values = locals().copy()
    if not (any((isinstance(_prefix_values.copy().get(key), dict) and _prefix_values.copy().get(key) for key in ('inputs_dict', 'input_dict', 'input_data'))) or (isinstance(_prefix_values.copy().get('data'), dict) and _prefix_values.copy()['data'].get('input_data'))):
        try:
            inputs_dict = {'k': k.cpu(), 'v': v.cpu(), 'beta': beta.cpu(), 'gk': gk.cpu(), 'A': A.cpu(), 'dk': dk.cpu(), 'dw': dw.cpu(), 'du': du.cpu(), 'dg': dg.cpu(), 'cu_seqlens': None, 'chunk_indices': None}
        except NameError:
            pass
    return locals()
