"""Existing independent CPU equations; file-generation main omitted."""
import types
triton = types.SimpleNamespace(cdiv=lambda n, d: (n + d - 1) // d, next_power_of_2=lambda n: 1 << (int(n) - 1).bit_length())
import torch

def make_inputs():
    torch.manual_seed(20261031)
    B = 2
    T = 1024
    H = 8
    HQ = 16
    K = 64
    V = 64
    S = 256
    BT = 64
    BS = 32
    num_blocks = triton.cdiv(T, S) + 1
    device = torch.device('cpu')
    dtype = torch.float32
    q = torch.randn(B, T, num_blocks, HQ, K, dtype=dtype, device=device)
    k = torch.randn(B, T, H, K, dtype=dtype, device=device)
    v = torch.randn(B, T, H, V, dtype=dtype, device=device)
    g_cumsum = torch.randn(B, T, HQ, dtype=dtype, device=device)
    do = torch.randn(B, T, HQ, V, dtype=dtype, device=device)
    dg_cumsum = torch.zeros(B, T, HQ, dtype=torch.float32, device=device)
    NS = triton.cdiv(T, S)
    hc_whole = torch.randn(B * NS, H, K, K, dtype=torch.float32, device=device)
    L = torch.randn(B, T, HQ, dtype=torch.float32, device=device)
    D = torch.randn(B, T, HQ, dtype=torch.float32, device=device)
    scale = 1.0 / K ** 0.5
    inputs_dict = {'q': q.cpu(), 'k': k.cpu(), 'v': v.cpu(), 'g_cumsum': g_cumsum.cpu(), 'do': do.cpu(), 'dg_cumsum': dg_cumsum.cpu(), 'hc_whole': hc_whole.cpu(), 'L': L.cpu(), 'D': D.cpu(), 'scale': scale, 'S': S, 'BT': BT, 'BS': BS}
    _prefix_values = locals().copy()
    if not (any((isinstance(_prefix_values.copy().get(key), dict) and _prefix_values.copy().get(key) for key in ('inputs_dict', 'input_dict', 'input_data'))) or (isinstance(_prefix_values.copy().get('data'), dict) and _prefix_values.copy()['data'].get('input_data'))):
        try:
            inputs_dict = {'q': q.cpu(), 'k': k.cpu(), 'v': v.cpu(), 'g_cumsum': g_cumsum.cpu(), 'do': do.cpu(), 'dg_cumsum': dg_cumsum.cpu(), 'hc_whole': hc_whole.cpu(), 'L': L.cpu(), 'D': D.cpu(), 'scale': scale, 'S': S, 'BT': BT, 'BS': BS}
        except NameError:
            pass
    return locals()
