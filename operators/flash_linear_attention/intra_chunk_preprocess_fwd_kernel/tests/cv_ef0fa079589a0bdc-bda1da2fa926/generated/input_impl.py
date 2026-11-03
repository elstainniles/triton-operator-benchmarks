"""Existing independent CPU equations; file-generation main omitted."""
import torch

def make_inputs():
    torch.manual_seed(20261031)
    B = 2
    T = 256
    H = 8
    HQ = 16
    K = 32
    V = 32
    BT = 32
    device = torch.device('cpu')
    dtype = torch.float32
    q = torch.randn(B, T, HQ, K, dtype=dtype, device=device)
    k = torch.randn(B, T, H, K, dtype=dtype, device=device)
    v = torch.randn(B, T, H, V, dtype=dtype, device=device)
    w = torch.randn(B, T, H, K, dtype=dtype, device=device)
    beta = torch.randn(B, T, H, dtype=dtype, device=device)
    g_cumsum = torch.randn(B, T, HQ, dtype=dtype, device=device)
    A = torch.randn(B, T, H, BT, dtype=dtype, device=device)
    scale = 1.0 / K ** 0.5
    inputs_dict = {'q': q.cpu(), 'k': k.cpu(), 'v': v.cpu(), 'w': w.cpu(), 'beta': beta.cpu(), 'g_cumsum': g_cumsum.cpu(), 'A': A.cpu(), 'scale': scale, 'BT': BT}
    _prefix_values = locals().copy()
    if not (any((isinstance(_prefix_values.copy().get(key), dict) and _prefix_values.copy().get(key) for key in ('inputs_dict', 'input_dict', 'input_data'))) or (isinstance(_prefix_values.copy().get('data'), dict) and _prefix_values.copy()['data'].get('input_data'))):
        try:
            inputs_dict = {'q': q.cpu(), 'k': k.cpu(), 'v': v.cpu(), 'w': w.cpu(), 'beta': beta.cpu(), 'g_cumsum': g_cumsum.cpu(), 'A': A.cpu(), 'scale': scale, 'BT': BT}
        except NameError:
            pass
    return locals()
