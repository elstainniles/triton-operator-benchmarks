"""Existing independent CPU equations; file-generation main omitted."""
import types
triton = types.SimpleNamespace(cdiv=lambda n, d: (n + d - 1) // d, next_power_of_2=lambda n: 1 << (int(n) - 1).bit_length())
import torch

def make_inputs():
    torch.manual_seed(20261031)
    B = 2
    T = 1024
    H = 8
    HQ = 8
    G = HQ // H
    K = 64
    V = 64
    BT = 64
    BS = 32
    BK = 16
    BV = 64
    scale = K ** (-0.5)
    device = torch.device('cpu')
    dtype = torch.float32
    q = torch.randn(B, T, HQ, K, dtype=dtype, device=device)
    k = torch.randn(B, T, H, K, dtype=dtype, device=device)
    v = torch.randn(B, T, H, V, dtype=dtype, device=device)
    lse = torch.randn(B, T, HQ, dtype=torch.float, device=device)
    delta = torch.randn(B, T, HQ, dtype=torch.float, device=device)
    do = torch.randn(B, T, HQ, V, dtype=dtype, device=device)
    dq = torch.zeros(B, T, HQ, K, dtype=dtype, device=device)
    dg_cumsum = None
    g_cumsum = None
    inputs_dict = {'q': q.cpu(), 'k': k.cpu(), 'v': v.cpu(), 'lse': lse.cpu(), 'delta': delta.cpu(), 'do': do.cpu(), 'dq': dq.cpu(), 'dg_cumsum': dg_cumsum, 'g_cumsum': g_cumsum, 'scale': scale, 'B': B, 'T': T, 'H': H, 'HQ': HQ, 'G': G, 'K': K, 'V': V, 'BT': BT, 'BS': BS, 'BK': BK, 'BV': BV}
    q_clone = q.clone()
    k_clone = k.clone()
    v_clone = v.clone()
    lse_clone = lse.clone()
    delta_clone = delta.clone()
    do_clone = do.clone()
    dq_clone = dq.clone()
    NV = triton.cdiv(V, BV)
    NT = triton.cdiv(T, BT)
    grid = (NV, NT, B * HQ)
    _prefix_values = locals().copy()
    if not (any((isinstance(_prefix_values.copy().get(key), dict) and _prefix_values.copy().get(key) for key in ('inputs_dict', 'input_dict', 'input_data'))) or (isinstance(_prefix_values.copy().get('data'), dict) and _prefix_values.copy()['data'].get('input_data'))):
        try:
            inputs_dict = {'q': q.cpu(), 'k': k.cpu(), 'v': v.cpu(), 'lse': lse.cpu(), 'delta': delta.cpu(), 'do': do.cpu(), 'dq': dq.cpu(), 'dg_cumsum': dg_cumsum, 'g_cumsum': g_cumsum, 'scale': scale, 'B': B, 'T': T, 'H': H, 'HQ': HQ, 'G': G, 'K': K, 'V': V, 'BT': BT, 'BS': BS, 'BK': BK, 'BV': BV}
        except NameError:
            pass
    return locals()
