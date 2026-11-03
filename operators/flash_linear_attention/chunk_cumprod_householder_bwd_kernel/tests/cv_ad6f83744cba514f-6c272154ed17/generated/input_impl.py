"""Existing independent CPU equations; file-generation main omitted."""
import types
triton = types.SimpleNamespace(cdiv=lambda n, d: (n + d - 1) // d, next_power_of_2=lambda n: 1 << (int(n) - 1).bit_length())
import torch

def make_inputs():
    torch.manual_seed(20261031)
    'Test function for chunk_cumprod_householder_bwd_kernel'
    B = 2
    T = 128
    HQ = 16
    H = 4
    K = 32
    S = 64
    BT = 32
    device = torch.device('cpu')
    dtype = torch.float32
    num_chunks = B * triton.cdiv(T, BT)
    num_splits = B * triton.cdiv(T, S)
    w1 = torch.randn(B, T, H, K, dtype=dtype, device=device)
    w2 = torch.randn(B, T, H, K, dtype=dtype, device=device)
    k = torch.randn(B, T, H, K, dtype=dtype, device=device)
    dk = torch.randn(B, T, HQ, K, dtype=dtype, device=device)
    hc_suffix = torch.randn(num_chunks, H, K, K, dtype=dtype, device=device)
    dhc_whole = torch.randn(num_splits, HQ, K, K, dtype=dtype, device=device)
    cu_seqlens = None
    inputs_dict = {'w1': w1.cpu(), 'w2': w2.cpu(), 'k': k.cpu(), 'dk': dk.cpu(), 'hc_suffix': hc_suffix.cpu(), 'dhc_whole': dhc_whole.cpu(), 'cu_seqlens': cu_seqlens, 'S': S, 'BT': BT}
    _prefix_values = locals().copy()
    if not (any((isinstance(_prefix_values.copy().get(key), dict) and _prefix_values.copy().get(key) for key in ('inputs_dict', 'input_dict', 'input_data'))) or (isinstance(_prefix_values.copy().get('data'), dict) and _prefix_values.copy()['data'].get('input_data'))):
        try:
            inputs_dict = {'w1': w1.cpu(), 'w2': w2.cpu(), 'k': k.cpu(), 'dk': dk.cpu(), 'hc_suffix': hc_suffix.cpu(), 'dhc_whole': dhc_whole.cpu(), 'cu_seqlens': cu_seqlens, 'S': S, 'BT': BT}
        except NameError:
            pass
    return locals()
